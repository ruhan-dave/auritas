# legal_assistant

### General 
I built a classic RAG system (with only the embedding and LLM as API sources) to provide a Question and Answer experience for users. 
This simple app uses Streamlit as the front end UI, whereby the user would input a pdf file, then the app will prompt the user to type in 
a question, based on which the LLM in the backend will chunk, embed, and retrieve the query and the documents, then provide an answer 

### Chunking
With each chunk size being 150 characters (basically a couple of short sentences), I split all the text parsed through the PDF file the user passes into the streamlit application.
These chunks are then embedded and stored into a vector database. 

### Embeddings
I went with Embeddings model from Cohere "embed-english-light-v3.0", which is a light-weight, but still capable embedding optimized for a few vector Databases including Qdrant. 
This embedding model is perfect for a demo like this application here. You can see other embedding models here: https://cohere.com/blog/introducing-embed-v3

### Vector Database 
I used Qdrant to store and store and retrieve document embeddings. Qdrant is a vector database designed to be scalable and reliable for 
storing vector objects. It also offers relatively straightforward deployment options, particularly that they work quite well with streamlit. 

### LLM
I decided to go with Cohere, which is a San-Francisco Based AI startup with plenty of models to choose from. This particular app makes API calls to the "command-r-08-2024"
model from Cohere. To make this work, I have input a template for the LLM to make reference to. The user query and the will be passed to the LLM to retrieve a final
output, which is then displayed in the Streamlit App. 


