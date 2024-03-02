"""
This contains a simple script to the pipeline
"""

from db_retriever import DBRetriever
from qa_inference import QA
from langchain.chains import RetrievalQA
import os
import transformers


def pipeline(api_key:str, question: str, hf_auth=hf_auth, device=-1, verbose=True) -> str:
	"""
	Create a pipeline for the question anwering
	:param config: configuration to set up for injector
	:param question: question as the query
	:param device: choose the device to run this pipeline on, for upgrading to GPU change this to  (default: -1, which means for CPU)
	:return: final prompt as output of KAPING to feed into a QA model
	"""


	# create retriever KGs from Pinecone database

	retriever = DBRetriever(api_key=api_key, 
							index_name="smartpub", namespace="assignment_embedding",
							model_name='meta-llama/Llama-2-13b-chat-hf', batch_size=32)

	docs_pmid = retriever.getTopSimilarDocs(question=question, num_docs=10)

	# QA model
	qa = QA(prompt=question, device=device, hf_auth=hf_auth)
	qa.qa_inference(qa.task, qa.model_name)

	rag_pipeline = RetrievalQA.from_chain_type(
	    llm=qa.llm,
	    chain_type="stuff",
	    verbose=verbose,
	    retriever=retriever.vectorstore_db.as_retriever(search_kwargs={"k":10}),
	    chain_type_kwargs={
	        "verbose": verbose },

	)

	return rag_pipeline

	

