import math
from collections import Counter
from tqdm import tqdm
import time

class BM25():
    def __init__(self, documents):
        """
        Initializes the BM25 instance.

        Parameters:
        - documents (list): A list of documents, where each document is represented as a list of strings.
        """
        self.documents = documents
        self.N = len(documents)
        self.avg_len_docs = sum(len(doc) for doc in documents) / self.N
        self.word_counts = self.calculate_word_counts()
        self.k1 = 1.2
        self.b = 0.75

        
        """print("Calcualting IDF Values")
        idf_start = time.time()
        self.idf_values = {word: self.get_idf(word) for word in tqdm(self.word_counts.keys())}
        idf_end = time.time()
        execution_time = idf_end - idf_start
        print(f"Execution time IDF: {execution_time} seconds")   """    


    def calculate_word_counts(self):
        word_counts = Counter()
        for document in self.documents:
            word_counts.update(document)
        return word_counts

    def rank_documents(self, query):
        """
        Ranks documents in the corpus with respect to a given query using Okapi BM25.

        Parameters:
        - query (list): The query represented as a list of strings.

        Returns:
        - ranked_documents (list): A list of tuples, where each tuple contains a document and its BM25 score.
        """
        scored_documents = []
        for doc in tqdm(self.documents, desc="Scoring Documents"):
            score = self.score_document(doc, query)
            scored_documents.append((doc, score))

        ranked_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
        return ranked_documents
    
    def score_document(self, document, query):
        """
        Computes the BM25 score for a document with respect to a given query.

        Parameters:
        - document (list): The document represented as a list of strings.
        - query (list): The query represented as a list of strings.

        Returns:
        - score (float): The BM25 score for the document.
        """
        score = 0.0
        doc_word_counts = Counter(document)

        for word in query:
            if word not in self.word_counts.keys():
                continue
            idf = self.calculate_idf(word)
            numerator = doc_word_counts[word] + (self.k1 + 1)
            denominator = doc_word_counts[word] + self.k1 * (1 - self.b + self.b * (len(document) / self.avg_len_docs))
            score += idf * (numerator / denominator)

        return score

    def calculate_idf(self, word):
        """
        Computes the IDF (Inverse Document Frequency) for a given word.

        Parameters:
        - word (str): The word for which IDF is computed.

        Returns:
        - idf (float): The IDF value for the word.
        """
        n_docs_w_word = sum(1 for doc in self.documents if word in doc)
        return math.log((self.N - n_docs_w_word + 0.5) / (n_docs_w_word + 0.5) + 1)

