"""
This contains a simple script to set up the connection to the Pinecone database that contains all of the documents
"""

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pinecone
#from langchain.vectorstores import Pinecone 
from langchain_pinecone import Pinecone
import torch
from transformers import LlamaTokenizer

class DBRetriever:

    def __init__(self, api_key:str, hf_auth:str, index_name:str="smartpub", 
                    model_name:str='sentence-transformers/all-MiniLM-L6-v2', batch_size=32, device=torch.device('cpu')):

        """
        This class uses a vectorstore as a retriever to be part of the QA pipeline to retrieve the most relevant documents
        Default model is the SentenceTransformer Mini Llama

        :param str api_key: The API key for accessing the service.
        :param str hf_auth: The HF authentication key to retrieve
        :param str index_name: The name of the index to be used (default is "smartpub").
        :param str model_name: The name of the pre-trained model to be used (default is 'sentence-transformers/all-MiniLM-L6-v2').
        :param int batch_size: The batch size for processing data (default is 32).
        :param torch.device device: The device to be used for processing (default is for CPU).

        
        """

        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=api_key) 

        # Initialize the existing index
        self.index = pc.Index(name=index_name)


        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf",token=hf_auth)    
        self.embed_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': batch_size}
            )
        self.vectorstore_db = Pinecone(self.index, self.embed_model.embed_query, 'relations')

