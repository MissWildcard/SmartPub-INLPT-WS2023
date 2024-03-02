''' Implementation to extract knowledge graph from the pubmed dataset.
Knowlegde Graph is extracted in the form of triplets (entity, relation, entity)
For this, we use Babelscape/rebel model from https://github.com/Babelscape/rebel '''

import pandas as pd
from transformers import pipeline, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize
import torch
import json

#CHANGE THE YEAR TO THE SPECIFIC YEAR YOU NEED
year=2014

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Name of output json file
out_file1= f'ER_{year}.json'
out_file2= f'ER_{year}_list.json'

# Initialize the triplet extraction pipeline using Babelscape/rebel-large model
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large', device=device)

# Load the tokenizer for the desired model
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
max_token_length = tokenizer.model_max_length

# Load the CSV file into a DataFrame
df_all = pd.read_csv("AllData.csv")
df = df_all[df_all['Year'] == year] 
print("Count of rows in df:", len(df))

# Function to process each row and generate triplets
def processRow(row, output_file1, output_file2):
    title = str(row['Title'])
    abstract = str(row['Abstract'])
    pmid = int(row['PMID'])  
    authors = row['Authors']
    date = f"{int(row['Month'])}-{int(row['Year'])}"
    print(pmid)
    
    # Initialize dictionaries to store entity relation information
    title_entity_relation = {}
    abstract_entity_relation = {}

    # Omit empty title or abstracts
    if not title.strip():
        if abstract.strip():
            abstract_selected = abstract
            triplets = genAbstractTextFromPipeline(abstract_selected)
            writeTripletsToFile(triplets, pmid, authors, date, output_file1, output_file1)
    elif not abstract.strip():
        if title.strip():
            title_selected = title
            triplets = genTitleTextFromPipeline(title_selected)
            writeTripletsToFile(triplets, pmid, authors, date, output_file1, output_file2)
    else:
        title_selected = title
        abstract_selected = abstract 
        triplets = genTitleTextFromPipeline(title_selected)
        writeTripletsToFile(triplets, pmid, authors, date, output_file1, output_file2)
        triplets = genAbstractTextFromPipeline(abstract_selected)
        writeTripletsToFile(triplets, pmid, authors, date, output_file1, output_file2)


def genAbstractTextFromPipeline(abs):
    # If abstract length exceeds the maximum token length, use sliding window approach for splitting
    if len(abs) > max_token_length:
        # Initialize an empty list to store the triplets
        triplets_list = []
        # Define the window size for sliding window approach
        window_size = max_token_length
        # Iterate over the input sequence with sliding window
        for i in range(0, len(abs), window_size):
            window_abs = abs[i:i+window_size]
            # Generate text using the triplet extraction pipeline for the current window
            generated_abs_text = triplet_extractor(window_abs, return_tensors=True, return_text=False)
            generated_abs_text_decoded = triplet_extractor.tokenizer.batch_decode([generated_abs_text[0]["generated_token_ids"]])[0]
            # Extract triplets from the generated text for the current window
            triplets = extractTriplets(generated_abs_text_decoded)
            # Append the triplets to the list
            triplets_list.extend(triplets)
        return triplets_list
    else:
        # Generate abstract text using the triplet extraction pipeline
        generated_abs_text = triplet_extractor(abs, return_tensors=True, return_text=False)
        generated_abs_text_decoded = triplet_extractor.tokenizer.batch_decode([generated_abs_text[0]["generated_token_ids"]])[0]
        # Extract and return triplets from the generated text
        return extractTriplets(generated_abs_text_decoded)

def genTitleTextFromPipeline(title):
    # Generate text using the triplet extraction pipeline
    generated_title_text = triplet_extractor(title, return_tensors=True, return_text=False)
    generated_title_text_decoded = triplet_extractor.tokenizer.batch_decode([generated_title_text[0]["generated_token_ids"]])[0]
    return extractTriplets(generated_title_text_decoded)

# Function to parse the generated text and extract the triplets
def extractTriplets(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    return triplets

def writeTripletsToFile(triplets, pmid, authors, date, out_1, out_2):
    #In this file, each extracted knowledge graph corresponds to an entry in the output file . This is for experimanetal purposes
    with open(out_1, 'a') as f:
        for triplet in triplets:
            f.write(f"Triplet: {triplet}, PMID: {pmid}, Authors: {authors}, Date: {date}\n")
    #In this file, all knowledge graph associated to a single pmid is stored as a single entry. This is for experimanetal purposes
    with open(out_2, 'a') as f:
        triplet_entry = {
            "title_entity_relation": {
                "pmid": pmid,
                "triplets": triplets,
                "authors": authors,
                "year": date
            },
            "abstract_entity_relation": {
                "pmid": pmid,
                "triplets": triplets,
                "authors": authors,
                "year": date
            }
        }
        f.write(json.dumps(triplet_entry) + '\n')

# Create lists to store processed data
title_entity_relations = []
abstract_entity_relations = []

# Process each row
for index, row in df.iterrows():
    processRow(row, out_file1, out_file2)
