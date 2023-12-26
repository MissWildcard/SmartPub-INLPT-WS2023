from BM25 import BM25 as BM25
import argparse
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle as pkl
import os

def main(args):

    if args.file_path:
        print("Initializing new Model. This may take some time.")                
        file_path = args.file_path

        if args.not_tokenised:
            filename = os.path.basename(file_path)
            corpus = pd.read_csv(file_path, encoding='utf-8')

            documents = corpus["Title"].astype(str) + ' ' + corpus["Abstract"].astype(str)
            tokenized_dump_path = f"data/{filename.remove('.csv')}_bm25_tok.pkl"
            print("Tokenizing Corpus:")
            tokenized_documents = []
            for doc in tqdm(documents):
                tokenized_documents.append(word_tokenize(doc.lower()))
            
            with open(tokenized_dump_path, 'wb') as file:
                pkl.dump(tokenized_documents, file)
            print(f"Tokenizing done. Tokenized Corpus saved at {tokenized_dump_path}")
        else:
            assert file_path.endswith(".pkl"), "File must have a '.pkl' extension. Please make sure to set --not_tokenised when using a new or untokenised corpus."
            with open(file_path, 'rb') as file:
                tokenized_documents = pkl.load(file)

        Scorer = BM25(tokenized_documents)
    elif args.model_path:
        print("Reusing Model")
        with open(args.model_path, 'rb') as file:
            Scorer = pkl.load(file)
    run = True
    while run:
        query = input("Please enter your query. To exit type 'exit': ")
        if query == "exit": 
            run = False
            continue

        tokenised_query = word_tokenize(query.lower())

        ranked_docs = Scorer.rank_documents(tokenised_query)
        print(f"Score: {ranked_docs[0][1]}\nAnswer: { ' '.join(ranked_docs[0][0])}")
        print(f"Score: {ranked_docs[1][1]}\nAnswer: { ' '.join(ranked_docs[1][0])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    exclusive_group = parser.add_mutually_exclusive_group(required=True)

    exclusive_group.add_argument('-f', '--file_path', action="store")
    exclusive_group.add_argument('-m', '--model_path', action="store")

    parser.add_argument("--not_tokenised", action='store_true') 
    args = parser.parse_args()
    main(args)