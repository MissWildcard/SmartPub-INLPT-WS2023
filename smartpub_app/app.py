from flask import Flask, render_template, jsonify, request 
from src.helper import *
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import os 
from model.model import pipeline
from model.model import pipeline
from model.qa_inference import QA
from langchain.chains import RetrievalQA
import os
import transformers


app = Flask(__name__)

def setup_pipeline():
    api_key = os.environ.get('PIPELINE_API_KEY')  # Replace with your API key
    hf_auth = os.environ.get('HF_AUTH') # Replace with your Hugging Face authentication token
    return api_key,hf_auth

# Initialize your pipeline when the application starts
api_key,hf_auth = setup_pipeline()

@app.route("/")
def index():
    return render_template('chat.html')

print("")
@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        print(input)
        result = pipeline(api_key= api_key,hf_auth=hf_auth,question=input)
        print("Response : ", result)
        return str(result)
    except Exception as e:
        print("An error occurred:", e)
        return "An error occurred: " + str(e), 500 




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)