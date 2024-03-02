import numpy as np
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class DBRetriever:

    def __init__(self, api_key:str="3c16547c-0c89-48d3-ac87-5ce64dfce92d", index_name:str="smartpub", namespace:str="assignment_embedding", 
                    model_name:str='sentence-transformers/all-MiniLM-L6-v2', batch_size=32):
        #If system supports cude, use it for gpu execution
        if torch.cuda.is_available(): 
            device = 'cuda:0'
        else: 
            device = 'cpu'
            print("You are working on CPU. This may lead to long computation times.")
        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=api_key) 

        # Initialize the existing index
        self.index = pc.Index(name=index_name)

        #Using pre-trained STS model
        self.sts_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': batch_size}
            )

    def getTopSimilarDocs(self, question:str="Tell me about Schizophrenia?", num_docs:int=5):

        question_embedding = self.sts_model.embed_documents([question])[0]
        
        # Calculate STS between question and each document
        #question_embedding = sts_model.embed_documents([question])[0]
        
        self.vectorstore_db = Pinecone(self.index, self.sts_model.embed_query, 'relations')
        # Query the index to find similar documents
        

        query_results = self.vectorstore_db.similarity_search(query=question, k=num_docs)
        # Extract the IDs or indices of the top similar documents
        

        return 
