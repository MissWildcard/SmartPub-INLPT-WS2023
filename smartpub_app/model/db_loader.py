import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import json
from pinecone import Pinecone as pc
from pinecone import PodSpec
from langchain_pinecone import Pinecone as vectorstore_pc
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import LlamaTokenizer
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import pandas as pd

_ = load_dotenv('../.env')

class PineconeVDB():
    """
    This class instantiates the Pinecone Vector-DataBase. It allows for the indexing of new documents as well as finding the most similar indexed document to a query.
    """
    def __init__(self, embedding_model_name: str, batch_size: int = 32):

        # throw exception if no GPU is available
        if torch.cuda.is_available(): 
            self.device = 'cuda:0'
        else: 
            self.device = 'cpu'
            print("You are working on CPU. This may lead to long computation times.")
            
        self.batch_size = batch_size

        hf_auth = os.getenv('HF_AUTH')
        #DEGUGGING LINES*2
        print('I am in dbloader...........')
        print(hf_auth)
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf",token=hf_auth)
        
        self.embedding_model_name = embedding_model_name     
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': batch_size}
        )

        self.pc_db = pc(
            api_key=os.environ.get('PINECONE_API_KEY') ,
            environment=os.environ.get('PINECONE_ENVIRONMENT')
        )


    def create_pinecone_index(self, index_name: str, metric: str = "cosine") -> pc.Index:
        doc = "This is a placeholder to get the embedding dimensions correct and should not be pushed to the index."
        embedding_dimension = len(self.embedding_model.embed_documents(doc)[0])
        
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
            print(f"Index {index_name} created.")
        else:
            print(f"Index {index_name} already exists")
        

    def push_data_to_index(self, index_name: str, data: list[dict], namespace:str):
        #if index_name not in self.pc_db.list_indexes().names():
        #    return ValueError("The Index you are trying to push to does not exist. Use PineconeVDB.list_indexes() to see the available indexes.")            
        #else:
        index = self.pc_db.Index(index_name)

        for i in tqdm(range(0, len(data), self.batch_size), desc=f"Pushing data to Index {index_name}:"):       
            i_end = min(len(data), i+ self.batch_size)
            batch = data[i:i_end]

            relations = []
            metadata = []
            ids = []

            for entry in batch:
                x = 'title_entity_relation'
                ids.append(str(entry[x]["pmid"]))
                authors = entry[x]["authors"]
                date = entry[x]["year"]

                title_relations = readout_triplets(entry[x]["triplets"])
                abstract_relations = readout_triplets(entry['abstract_entity_relation']["triplets"])
                relation = f"Title: {title_relations}, Abstract: {abstract_relations}"
                relations.append(relation)

                # get metadata to store in Pinecone
                metadata.append({'relations': relation,
                                   'authors': authors,
                                   'date': date
                                   })
            
            embeddings = self.embedding_model.embed_documents(relations)
                
            index.upsert(vectors=zip(ids, embeddings, metadata), namespace=namespace)
        print("Indexing Complete!")

    def find_most_similar_docs(self, query: str, index: pc.Index, num_of_chunks: int = 3):
        vectorstore = vectorstore_pc(index, self.embedding_model, 'relations')
        return vectorstore.similarity_search(
                query, 
                k=num_of_chunks 
                )
    
    def show_index_info(self, index_name):        
        print(self.pc_db.describe_index(index_name))
    
    def list_indexes(self):
        print(self.pc_db.list_indexes())

    def delete_index(self, index_name):
        self.pc_db.delete_index(index_name)
        print(f"Deleted Index {index_name}")

def readout_triplets(triplets: list[dict]):
    readout = ""
    for triplet in triplets:
        readout += f"|{triplet['head']} - {triplet['type']} - {triplet['tail']}| "
    return readout

class PineconeVDBRawText(PineconeVDB):
    """
    This class inherits the PineconeVDB, doing the same purpose, but use the raw text as content instead the KGs
    """
    def __init__(self, embedding_model_name: str, batch_size: int = 32):
        super().__init__(embedding_model_name, batch_size)
        self.create_text_splitter()

    def create_text_splitter(self):
        token_len = lambda text: len(self.tokenizer.encode(text))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=token_len,
            separators=['\n\n', '\n', ' ', '']
        ) 

    def push_data_to_index(self, index_name:str, data:pd.DataFrame):
        index = self.pc_db.Index(index_name)

        for i in tqdm(range(0, len(data), self.batch_size), desc=f"Pushing data to Index {index_name}:"):       
            i_end = min(len(data), i+ self.batch_size)
            batch = data[i:i_end]

            texts = []
            metadata = []
            ids = []

            for entry in batch:
                
                # take care of title or abstract that is empty
                pmid = str(row['PMID'])
                title = str(row['Title']).strip()
                abstract = str(row['Abstract']).strip()
                authors = row['Authors']
                date = f"{int(row['Month'])}-{int(row['Year'])}"

                if not title:
                    if abstract:
                        ids.append(pmid)
                        texts.append('text:' f'Title:  Abstract: {abstract}')
                elif not abstract:
                    if title:
                        ids.append(pmid)
                        texts.append(f'Title: {title} Abstract:')
                else:
                    chunks = self.text_splitter.split_text(row.Abstract)
                    title = row.Title
                    for i, chunk in enumerate(chunks):
                        ids.append(f'{pmid}-{i}')
                        texts.append(f'Title: {title} Abstract p{i+1} {chunk}')
                

                # get metadata to store in Pinecone
                metadata.append({'text': texts,
                                   'authors': authors,
                                   'date': date
                                   })
            
            embeddings = self.embedding_model.embed_documents(texts)
                
            index.upsert(vectors=zip(ids, embeddings, metadata), namespace=namespace)
        print("Indexing Complete!")



if __name__ == 'main':
    # directory_path = "smartpub_app\model\data_files"
    # db = PineconeVDB(embedding_model_name='sentence-transformers/all-MiniLM-L6-v2')
    # index_name = "smartpub"
    # #db.create_pinecone_index(index_name)
    # namespace = "assignment_embedding"
    
    # for filename in os.listdir(directory_path):
    #     file_path = os.path.join(directory_path, filename)
    #     if os.path.isfile(file_path):
    #         with open(file_path, 'r', encoding='utf8') as file:
    #             print(f"Processing {filename}")
    #             data = json.load(file)


    #         db.push_data_to_index(index_name, data, namespace)
    #         print(f"Pushed {filename} to Index {index_name}")

    directory_path = sys.argv[1]
    db = PineconeVDBRawText(embedding_model_name='sentence-transformers/all-MiniLM-L6-v2')
    index_name = "smartpub"
    #db.create_pinecone_index(index_name)
    namespace = "assignment_rawtext"
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            print(f"Processing {filename}")
            data = pd.read_csv(file_path)


            db.push_data_to_index(index_name, data, namespace)
            print(f"Pushed {filename} to Index {index_name}")
