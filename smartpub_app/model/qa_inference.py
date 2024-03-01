"""
This contains the script to proceed the QA inference with some models
"""


from transformers import pipeline, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


class QA:
	
	def __init__(self, prompt: str, task:str="text-generation", model_name:str='meta-llama/Llama-2-13b-chat-hf',  device=-1):
		self.prompt = prompt
		self.task = task
		self.model_name = model_name
		self.device = device


	def qa_inference(self, prompt: str, task:str="text-generation", model_name:str='meta-llama/Llama-2-13b-chat-hf',  device=-1):
		"""
		Only use pretrained model (without any extra finetuning on any dataset)
		The idea is to pass a whole sequence into the pipeline in form of 
		 "<prompt> question: <question> answer: ""

		Tasks used are text generations/text2text generations depending on how experimented models are supported
		on Hugging Face

		This requires model to learn from input to continue generate text that is suitable for given prompt as input
		
		:param task: task for inference: text2text-generation / text-generation
		:param model_name: model used to inference: bert-large-uncased, t5-small, t5-base, t5-large, gpt2
		:param prompt: input prompt
		:param device: to use gpu or cpu (default is CPU, if using GPU, change to positive value from 0)
		:return: generated answer
		"""

		# for  meta-llama/Llama-2-13b-chat-hf
		# In this inference, as huggingface pipeline does not support text2text-generation for gpt2,
		# hence text-generation was used instead
		if task == "text-generation":
			bitsAndBites_config = transformers.BitsAndBytesConfig(
									    load_in_4bit=True,
									    bnb_4bit_quant_type='nf4',
									    bnb_4bit_use_double_quant=True,
									    bnb_4bit_compute_dtype=bfloat16
									)
			model_config = AutoConfig.from_pretrained(model_name, use_auth_token=os.environ.get('HF_AUTH'))
			model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
														config=model_config, quantization_config=bitsAndBites_config,
														device_map='auto', token=os.environ.get('HF_AUTH'))
			model.eval()
			self.qa_pipeline = pipeline(task, model=model_name, temperature=0.01, max_new_tokens=512, repetition_penalty=1.1, device=device)
			self.llm = HuggingFacePipeline(pipeline=qa_pipeline)
			answer = qa_pipeline(prompt)			
			return answer[0]['generated_text'].split('Answer: ', 1)[-1]

		# for bert-large-uncased, t5-small, t5-base, t5-large
		elif task == "text2text-generation":
			print("This task is used for bert-large-uncased and t5 models")
			qa_pipeline = pipeline(task, model=model_name, device=device)
			answer = qa_pipeline(prompt)
			return answer[0]['generated_text']






		





