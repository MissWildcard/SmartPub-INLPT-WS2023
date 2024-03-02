import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as pc
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


def setup():
    #If system supports cuda, use it for gpu execution
    if torch.cuda.is_available(): 
        device = 'cuda:0'
    else: 
        device = 'cpu'
        print("You are working on CPU. This may lead to long computation times.")
    # Initialize Pinecone client
    #pinecone = Pinecone(api_key= os.environ.get('HF_AUTH'))
    pinecone = Pinecone(api_key= "3c16547c-0c89-48d3-ac87-5ce64dfce92d")
    # Specify the name of the existing index
    index_name = "smartpub"
    namespace= "assignment_embedding"
    # Initialize the existing index
    index = pinecone.Index(name=index_name)

    #Using pre-trained STS model
    sts_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device},
            encode_kwargs={'device': device, 'batch_size': 32}
        )
    return index, sts_model, namespace

def getTopSimilarDocs(question = "Tell me about Schizophrenia?", num_docs=5, method=1):        
    index, sts_model, namespace= setup()
    # Calculate STS between question and each document
    question_embedding = sts_model.embed_documents([question])[0]
    vectorstore_db = pc(index, sts_model.embed_query, 'relations')
    #Direct STS method
    if method == 0:
        # Query the index to find similar documents
        query_results = vectorstore_db.similarity_search(query=question, k=num_docs, namespace=namespace)
        #Format the output    
        top_docs = []
        for result in query_results:
            content = result.page_content
            authors = result.metadata['authors']
            published = result.metadata['date']
            # Construct a dictionary containing the document content, metadata, and PMID
            doc_info = {
                'Content': content,
                'Authors': authors,
                'Date Published': published
            }
            top_docs.append(doc_info)
        return top_docs
    #STS using BERT
    elif method == 1:
        # Initialize pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        # Initialize an empty list to store similarity scores and document indices
        similarity_scores = []
        #Preliminary Similarity check
        query_results = index.query(vector=question_embedding, top_k=num_docs*5, namespace=namespace, include_metadata=True)
        #More finer level of similarity check among the preliminary results
        for match in query_results['matches']:
            # Extract document ID, score, and values
            score = match['score']
            text = match['metadata']['relations']
            authors = match.metadata['authors']
            date = match['metadata']['date']
            pmid = match['id']
            # Tokenize and encode query and document relation text
            ques_tokens = tokenizer.encode(question, add_special_tokens=True)
            document_tokens = tokenizer.encode(text, add_special_tokens=True)
            # Convert tokens to tensors
            ques_tensor = torch.tensor(ques_tokens).unsqueeze(0)  # Batch dimension
            document_tensor = torch.tensor(document_tokens).unsqueeze(0)  # Batch dimension
            #Get BERT embeddings for query and document relation text
            with torch.no_grad():
                ques_embedding = model(ques_tensor)[0][:, 0, :]  # First token (CLS token) pooled output
                document_embedding = model(document_tensor)[0][:, 0, :]  # First token (CLS token) pooled output
            doc_info = {
                'PMID' : pmid,
                'Content': text,
                'Authors': authors,
                'Date Published': date
            }
            # Compute cosine similarity between query and document relation embeddings
            similarity_score = cosine_similarity(ques_embedding, document_embedding)[0][0]
            # Store the similarity score and document relation text
            similarity_scores.append((doc_info, similarity_score))
        # Sort the document relations based on similarity scores (descending order)
        sorted_documents = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Retrieve the top 5 similar document relation texts
        top_docs = sorted_documents[:num_docs]
        return top_docs


#Sample usage
question = "Tell me about Schizophrenia?"
top_similar_docs = getTopSimilarDocs(question, method=1)
print(top_similar_docs)
