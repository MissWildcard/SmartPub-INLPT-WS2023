import torch
import transformers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import pinecone
from pinecone import Pinecone, PodSpec
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import LlamaTokenizer
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class EmbeddingModel():
    def __init__(self, embedding_model_name: str, batch_size: int = 32):

        # throw exception if no GPU is available
        if torch.cuda.is_available(): 
            self.device = 'cuda:0'
        else: 
            self.device = 'cpu'
            print("You are working on CPU. This may lead to long computation times.")
            
        self.batch_size = batch_size

        hf_auth = os.environ.get('HF_AUTH')
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf",token=hf_auth)
        
        self.embedding_model_name = embedding_model_name #        
        self.embed_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': batch_size}
        )


    def create_pinecone_index(self, index_name: str, metric: str = "cosine") -> pinecone.Index:
        doc = "This is a placeholder to get the embedding dimensions correct and should not be pushed to the index."
        embedding_dimension = len(self.embed_model.embed_documents(doc)[0])

        self.pc_db = Pinecone(
            api_key=os.environ.get('PINECONE_API_KEY') ,
            environment=os.environ.get('PINECONE_ENVIRONMENT')
        )

        if index_name not in self.pc_db.list_indexes().names():
            self.pc_db.create_index(
                index_name,
                dimension=embedding_dimension,
                metric=metric,
                spec=PodSpec(
                    environment='gcp-starter',
                    pod_type='s1.x1'
                )
            )
        return self.pc_db.Index(index_name)

    def push_data_to_index(self, index: pinecone.Index, data: pd.DataFrame):
        for i in tqdm(range(0, len(data), self.batch_size), desc=f"Pushing Data to Index {index}:"):       
            i_end = min(len(data), i+ self.batch_size)
            batch = data.iloc[i:i_end]
            ids = [x["PMID"] for i, x in batch.iterrows()]
            entities = [x["Entities"] for i, x in batch.iterrows()]
            embed_text = self.embed_model.embed_documents(entities)
            # get metadata to store in Pinecone
            metadata = [
                {'entities': entities,
                'authors': x['Authors'],
                'year': x['Year'],
                'month': x['Month']} for i, x in batch.iterrows()
            ]
            index.upsert(vectors=zip(ids, embed_text, metadata))
        print("Indexing Complete!")

    def find_most_similar_docs(self, query: str, index: Pinecone.Index, num_of_chunks: int = 3):
        vectorstore = Pinecone(index, self.embed_model.embed_query, 'text')
        return vectorstore.similarity_search(
                query, 
                k=num_of_chunks 
                )
    
    def show_index_info(self, index_name):        
        return self.pc_db.describe_index(index_name)
    
    def list_indexes(self):
        return self.pc_db.list_indexes()

    def delete_index(self, index_name):
        self.pc_db.delete_index(index_name)

if __name__ == '__main__':
    file_path = "test_corpus.csv"
    data= pd.read_csv(file_path)

    embedding_model = EmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')
    index = embedding_model.create_pinecone_index('test-index')
    #embedding_model.push_data_to_index(index, data)

    print(index.describe_index_stats())

    query = "How much learning data is needed for AI?"
    vectorstore = embedding_model.find_most_similar_docs(query)