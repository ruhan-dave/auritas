import streamlit as st
import cohere
from classic_rag import process_pdf_to_text, customize_chunking, retrieve_top_chunks, get_llm_output, query_chunking, no_pdf_retrieve_top_chunks, direct_answer_from_qdrant
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
import asyncio
from qdrant_client import QdrantClient
import numpy as np

# Set up asyncio event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Access API Key from Streamlit Secrets
cohere_key = st.secrets["cohere"]["api_key"] 
qdrant_key = st.secrets["qdrant"]["key"]
qdrant_host = st.secrets["qdrant"]["host"]

ch = cohere.ClientV2(api_key=cohere_key)

qdrant_client = QdrantClient(
    url=qdrant_host,
    prefer_grpc=True,  # Use gRPC for better performance
    api_key=qdrant_key,
    timeout=50
)

def extract_page_text(args):
    """Helper function for parallel PDF page extraction."""
    reader, page_num = args
    try:
        page = reader.pages[page_num]
        return " ".join(page.extract_text().split())  # Remove newlines and extra spaces
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")  # Error handling
        return ""

def retrieve_chunks_from_qdrant(user_query, collection_name="auritas", limit=5):
    """
    Retrieve relevant text chunks from Qdrant based on semantic similarity to the query.

    Args:
        user_query (str): The user's question
        collection_name (str): Name of the Qdrant collection to query
        limit (int): Number of results to retrieve

    Returns:
        list: List of relevant text chunks
    """
    try:
        # Generate embedding for the query
        embedding_response = ch.embed(
            texts=[user_query],
            model="embed-english-light-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        query_embedding = embedding_response.embeddings.float[0]

        # Search Qdrant directly using the query embedding
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )

        # Extract the text chunks from the search results
        retrieved_chunks = []
        for result in search_results:
            chunk_text = result.payload.get("text", "")
            if chunk_text:
                retrieved_chunks.append({
                    "text": chunk_text,
                    "score": result.score
                })

        return retrieved_chunks

    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return []

def get_llm_output(top_chunks, ch, query):

    # Create a clear prompt structure
    prompt = f"""Context information:
    {top_chunks}

    Based ONLY on the above context, please answer this question: {query}
    If the information isn't in the context, say "I don't have enough information to answer that question."
    """

    try:
        # Simple approach: include context directly in the message
        response = ch.chat(
            model="command-r-08-2024",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.message.content[0].text
    except Exception as e:
        print(f"Error in LLM response: {str(e)}")
        return f"Error generating response: {str(e)}"
    
def retrieve_answer_from_cohere(user_query, retrieved_chunks):
    preamble = """
    ## Task & Context
    You give answers to user's question with precision, based on the text string you receive.
    You should focus on serving the user's needs as best you can, which can be wide-ranging but always relevant to the document string.
    If you are not sure about the answer, you can ask for clarification or provide a general response saying you are not sure.
    
    ## Style Guide
    Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
    """
    # Generate response using Cohere V2 client with correct parameters
    documents = []
    for i, chunk in enumerate(retrieved_chunks):
        documents.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "data": {
                "title": f"Document Chunk {i}",
                "source": "Document"
            }
        })
    response = ch.chat(
        model="command-r-08-2024",
        messages=[{"role": "system", "content": preamble},
                  {"role": "user", "content": user_query}],
        documents=documents,  
        temperature=0.2
    )

        # tools=[{
        #     "name": "retrieval",
        #     "description": "Retrieve information from documents",
        #     "parameter_definitions": {},
        #     "returns": {"type": "string"},
        #     "tool_choice": "auto"
        # }]
        # connectors=[{
        #     "id": "retrieval",
        #     "documents": [{"text": chunk} for chunk in retrieved_chunks]
        # }]
        # system_prompt="You are an assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so."

    return response.message.content[0].text


def main():
    st.title("Hi, I'm Your Auritas Assistant")
    st.write("Ask me questions about our documents!")

    # Option to upload PDF or just ask questions
    tab1, tab2 = st.tabs(["Ask Questions", "Upload New Document"])

    with tab1:
        user_query = st.text_input("What would you like to know?")
        if user_query:
            with st.spinner("Searching for answers..."):
                top_chunks = retrieve_chunks_from_qdrant(user_query, collection_name="auritas", limit=5)
                ans = get_llm_output(top_chunks, ch, user_query)
                # top_chunks = retrieve_top_chunks(query=user_query, 
                #                  collection_name="auritas", 
                #                  chunks=list_chunks,
                #                  n=5)
                # answer = get_llm_output(top_chunks, ch, user_query)
                st.write(ans)

    with tab2:
        # Your existing PDF upload code here
        pdffile = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdffile is not None:
            st.write("Processing...")
            try:
                pdf_reader = PdfReader(pdffile)
                num_pages = len(pdf_reader.pages)
                st.write(f"Extracted {num_pages} pages from the PDF file.")
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
                return  # Exit function early if PDF cannot be processed
            
            # Process PDF text
            with ThreadPoolExecutor(max_workers=4) as executor:
                pdftext = list(executor.map(extract_page_text, [(pdf_reader, i) for i in range(num_pages)]))
                pdftext = " ".join(pdftext)
            
            st.write(pdftext)

            list_chunks = customize_chunking(pdftext)

            st.write(f"Generated {len(list_chunks)} text chunks from the PDF.")
            if not list_chunks:
                st.error("No text chunks were generated from the PDF.")
                return
            
            # Accept user query
            user_query = st.text_input("Ask a question about the document:")
            if user_query:
                st.write("Thinking...")

                # Retrieve relevant chunks
                top_chunks = retrieve_top_chunks(
                    client=client,
                    query=user_query, 
                    collection_name="new-collection", 
                    list_chunks=list_chunks, 
                    n=5
                )

                results = [chunk for chunk in top_chunks]
        
                st.write(f"Found {len(results)} relevant chunks: {results}")

                if not top_chunks:
                    st.error("No relevant information found. Try rephrasing your query.")
                    return

                # Get LLM output
                response = get_llm_output(top_chunks, ch, user_query)
                st.success("Here's what you need to know:")
                st.write(response)

if __name__ == "__main__":
    main()
