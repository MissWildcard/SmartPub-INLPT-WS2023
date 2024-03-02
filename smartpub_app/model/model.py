"""
This contains a simple script to the pipeline
"""

from db_retriever import DBRetriever
from qa_inference import QA
from langchain.chains import RetrievalQA
import os
import transformers


def pipeline(config, question: str, device=-1, verbose=True) -> str:
	"""
	Create a pipeline for the question anwering
	:param config: configuration to set up for injector
	:param question: question as the query
	:param device: choose the device to run this pipeline on, for upgrading to GPU change this to  (default: -1, which means for CPU)
	:return: final prompt as output of KAPING to feed into a QA model
	"""


	# create retriever KGs from Pinecone database

	retriever = DBRetriever(api_key=config.api_key, 
							index_name=config.index_name, namespace=config.namespace,
							model_name=config.model_name, batch_size=config.batch_size)

	docs_pmid = retriever.getTopSimilarDocs(question=question, num_docs=config.k)

	# QA model
	qa = QA(prompt=question, device=device, hf_auth=config.hf_auth)
	qa.qa_inference(qa.task, qa.model_name)

	rag_pipeline = RetrievalQA.from_chain_type(
	    llm=qa.llm,
	    chain_type="stuff",
	    verbose=verbose,
	    retriever=retriever.vectorstore_db.as_retriever(search_kwargs={"k":config.k}),
	    chain_type_kwargs={
	        "verbose": verbose },

	)

	answer = rag_pipeline(question)['result']

	return answer

	

