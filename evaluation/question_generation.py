import spacy
import re
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class InstanceCollector:
    def __init__(self, pubmed_corpus_file):
        self.nlp = spacy.load("en_core_web_sm")
        self.pubmed_corpus = pd.read_csv(pubmed_corpus_file)
        self.instances = []

    def collect_instances(self):
        for _, row in self.pubmed_corpus.iterrows():
            title = row['Title']
            if pd.notna(row['Abstract']):
                abstract = row['Abstract']                
            else:
                continue
            
            if self.has_pos_structure(title):
                if self.check_for_conclusive_part(abstract):
                    answer_long = self.extract_conclusive_part(abstract)
                    answer_long = self.preprocess_conclusion(answer_long)

                    question = self.convert_to_question(self.preprocess_title(title)).replace("[", "").replace("]", "")
                    answer_short = "No" if self.check_for_negation(title) else "Yes"

                    self.instances.append({
                        'question': question,
                        'short answer': answer_short,
                        'long answer': answer_long
                    })

    def has_pos_structure(self, text):
        tagged_tokens = pos_tag(word_tokenize(text))
        chunk_pattern = r'M: {<DT>?<JJ>*<NN.*>+}(<VBZ|VBP>)+?'  # NP followed by optional VBZ or VBP
        chunk_parser = RegexpParser(chunk_pattern)
        chunk_tree = chunk_parser.parse(tagged_tokens)
        return any(subtree.label() == 'M' for subtree in chunk_tree.subtrees())

    def extract_conclusive_part(self, abstract):
        start = abstract.lower().find('conclusion:')
        return abstract[start:] if start != -1 else ""
    
    def check_for_conclusive_part(self, abstract):
        if 'conclusion:' in abstract.lower(): return True
        else: return False

    def convert_to_question(self, title):
        tagged_tokens = pos_tag(word_tokenize(title.lower()))
        verb_token = next((word for word, pos in tagged_tokens if pos in ['vbz', 'vbp']), None)

        if verb_token == 'vbz':
            question = f"Is {title}?"
        elif verb_token == 'vbp':
            question = f"Are {title}?"
        else:
            question = f"Does {title}?"

        return question
    
    def preprocess_title(self, title):
        if title[-1] == ".":
            title = title[:-1]

        if ":" in title:
            ind = title.find(':')
            return title[:ind]
        else:
            return title
        
    def preprocess_conclusion(self, conclusion):
        conclusion = conclusion.lower().replace("conclusion:", "")
        return conclusion
        
        
    def check_for_negation(self, text):
        negations = ["not", "doesn't", "can't", "won't", "haven't"]
        pattern = re.compile(r'\b' + '|'.join(map(re.escape, negations)) + r'\b', re.IGNORECASE)
        return bool(re.search(pattern, text))
    

# Example usage
pubmed_corpus_file = "intelligence2023.csv"
collector = InstanceCollector(pubmed_corpus_file)
collector.collect_instances()

# Access the collected instances
with open('output.txt', 'w',  encoding='utf-8') as file:
    for instance in collector.instances:
        file.write(f"Question: {instance['question']}\n")
        file.write(f"Short Answer: {instance['short answer']}\n")
        file.write(f"Long Answer: {instance['long answer']}\n\n")

    file.write(f"Total instances: {len(collector.instances)}\n")