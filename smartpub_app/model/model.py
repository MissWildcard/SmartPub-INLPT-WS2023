"""
This contains a simple script to the pipeline
"""

from .db_retriever import DBRetriever
from .qa_inference import QA
from langchain.chains import RetrievalQA
import os
import transformers
import torch


def pipeline(api_key:str, question: str, hf_auth:str, 
			index_name="smartpub", model_name='meta-llama/Llama-2-13b-chat-hf',
			device=torch.device('cpu'), verbose=True, batch_size=32, k=10) -> str:
	"""
	Create a pipeline for the question anwering
	:param config: configuration to set up for injector
	:param question: question as the query
	:param device: choose the device to run this pipeline on, for upgrading to GPU change this to  (default: -1, which means for CPU)
	:return: final prompt as output of KAPING to feed into a QA model
	"""


	# create retriever KGs from Pinecone database

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	retriever = DBRetriever(api_key=api_key, 
							index_name=index_name,
							model_name=model_name, batch_size=batch_size, device=device)

	#docs_pmid = retriever.getTopSimilarDocs(question=question, num_docs=10)

	# QA model
	qa = QA(prompt=question, device=device, hf_auth=hf_auth)
	qa.qa_inference(qa.task, qa.model_name)

	rag_pipeline = RetrievalQA.from_chain_type(
	    llm=qa.llm,
	    chain_type="stuff",
	    verbose=verbose,
	    retriever=retriever.vectorstore_db.as_retriever(search_kwargs={"k":k}),
	    chain_type_kwargs={
	        "verbose": verbose },

	)

	answer = rag_pipeline['result']

	return answer

	

