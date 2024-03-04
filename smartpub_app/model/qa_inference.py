"""
This contains the script to proceed the QA inference with some models
"""


from transformers import pipeline, LlamaTokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from torch import bfloat16
from langchain.llms import HuggingFacePipeline
import os
import torch



class QA:
	
	def __init__(self, prompt: str, task:str="text-generation", model_name:str='meta-llama/Llama-2-13b-chat-hf',
				  device=torch.device('cpu'), hf_auth=os.environ.get('HF_AUTH')):
		self.prompt = prompt
		self.task = task
		self.model_name = model_name
		self.device = device
		self.hf_auth = hf_auth


	def qa_inference(self, task:str="text-generation", model_name:str='meta-llama/Llama-2-13b-chat-hf',  device=torch.device('cpu')):
		"""
		Only use pretrained model (without any extra finetuning on any dataset)

		Tasks used are text generations/text2text generations depending on how experimented models are supported
		on Hugging Face

		This requires model to learn from input to continue generate text that is suitable for given prompt as input
		
		:param task: task for inference: text2text-generation / text-generation
		:param model_name: model used to inference: meta-llama/Llama-2-13b-chat-hf, bert-large-uncased, t5-small, t5-base, t5-large
		:param device: to use gpu or cpu (default is CPU, if using GPU, change to positive value from 0)
		:return: generated answer / only in case of text2text-generation
		"""

		# for text-generation meta-llama/Llama-2-13b-chat-hf was used instead
		if task == "text-generation":
			tokenizer = LlamaTokenizer.from_pretrained(model_name, token=self.hf_auth)
			# bitsAndBites_config = BitsAndBytesConfig(
			# 						    load_in_4bit=True,
			# 						    bnb_4bit_quant_type='nf4',
			# 						    bnb_4bit_use_double_quant=True,
			# 						    bnb_4bit_compute_dtype=bfloat16
			# 						)
			model_config = AutoConfig.from_pretrained(model_name, token=self.hf_auth)
			model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
														config=model_config, #quantization_config=bitsAndBites_config,
														device_map='auto', token=self.hf_auth)
			model.eval()
			self.qa_pipeline = pipeline(task=task, tokenizer=tokenizer, model=model_name, temperature=0.01, max_new_tokens=512, repetition_penalty=1.1, device=device)
			self.llm = HuggingFacePipeline(pipeline=self.qa_pipeline)

		# for bert-large-uncased, t5-small, t5-base, t5-large
		elif task == "text2text-generation":
			print("This task is used for bert-large-uncased and t5 models")
			qa_pipeline = pipeline(task, model=model_name, device=device)
			answer = qa_pipeline(self.prompt)
			return answer[0]['generated_text']






		





