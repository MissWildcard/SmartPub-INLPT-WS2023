from flask import Flask, render_template, jsonify, request 
from dotenv import load_dotenv
from model.model import pipeline
import os
from model.db_retriever import DBRetriever

_ = load_dotenv('.env')

app = Flask(__name__)

def setup_pipeline():
    api_key = os.getenv('PINECONE_API_KEY')  # Replace with your API key
    hf_auth = os.getenv('HF_AUTH') # Replace with your Hugging Face authentication token
    print(api_key,hf_auth)
    return api_key,hf_auth

# Initialize your pipeline when the application starts
api_key,hf_auth = setup_pipeline()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input = msg
        print(input)
        retriever = DBRetriever(api_key, hf_auth)
        result = retriever.run(input)
        print("Response : ", result)
        return str(result)
    except Exception as e:
        print("An error occurred:", e)
        return "An error occurred: " + str(e), 500 




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)