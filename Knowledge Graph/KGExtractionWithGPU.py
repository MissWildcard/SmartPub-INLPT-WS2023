import pandas as pd
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
import torch

#CHANGE THE YEAR TO THE SPECIFIC YEAR YOU NEED
year=2013

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the triplet extraction pipeline using Babelscape/rebel-large model
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large', device=device)

# Load the CSV file into a DataFrame
df_all = pd.read_csv("AllData.csv")
df = df_all[df_all['Year'] == year] 
print("Count of rows in df:", len(df))

# Function to process each row and generate triplets
def process_row(row):
    title = str(row['Title'])
    abstract = str(row['Abstract'])
    pmid = int(row['PMID'])  
    authors = row['Authors']
    date = f"{int(row['Month'])}-{int(row['Year'])}"
    
    # Initialize dictionaries to store entity relation information
    title_entity_relation = {}
    abstract_entity_relation = {}

    # Omit empty title or abstracts
    if not title.strip():
        if abstract.strip():
            abstract_selected = abstract
            abstract_entity_relation['pmid'] = pmid
            abstract_entity_relation['triplets'] = genAbstractTextFromPipeline(abstract_selected)
            abstract_entity_relation['authors'] = authors
            abstract_entity_relation['year'] = date
            return pd.Series([None, abstract_entity_relation])
    elif not abstract.strip():
        if title.strip():
            title_selected = title
            title_entity_relation['pmid'] = pmid
            title_entity_relation['triplets'] = genTitleTextFromPipeline(title_selected)
            title_entity_relation['authors'] = authors
            title_entity_relation['year'] = date
            return pd.Series([title_entity_relation, None])
    else:
        title_selected = title
        abstract_selected = abstract 
        title_entity_relation['pmid'] = pmid
        title_entity_relation['triplets'] = genTitleTextFromPipeline(title_selected)
        title_entity_relation['authors'] = authors
        title_entity_relation['year'] = date
        
        abstract_entity_relation['pmid'] = pmid
        abstract_entity_relation['triplets'] = genAbstractTextFromPipeline(abstract_selected)
        abstract_entity_relation['authors'] = authors
        abstract_entity_relation['year'] = date
        return pd.Series([title_entity_relation, abstract_entity_relation])
    return None, None


def genAbstractTextFromPipeline(abs):
    # Generate text using the triplet extraction pipeline
    generated_abs_text = triplet_extractor(abs, return_tensors=True, return_text=False)
    generated_abs_text_decoded = triplet_extractor.tokenizer.batch_decode([generated_abs_text[0]["generated_token_ids"]])[0]
    # Extract triplets from the generated text
    return extract_triplets(generated_abs_text_decoded)

def genTitleTextFromPipeline(title):
    # Generate text using the triplet extraction pipeline
    generated_title_text = triplet_extractor(title, return_tensors=True, return_text=False)
    generated_title_text_decoded = triplet_extractor.tokenizer.batch_decode([generated_title_text[0]["generated_token_ids"]])[0]
    return extract_triplets(generated_title_text_decoded)

# Function to parse the generated text and extract the triplets
def extract_triplets(text):
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

# Create lists to store processed data
title_entity_relations = []
abstract_entity_relations = []

# Process each row
for index, row in df.iterrows():
    title_entity_relation, abstract_entity_relation = process_row(row)
    title_entity_relations.append(title_entity_relation)
    abstract_entity_relations.append(abstract_entity_relation)

# Create a DataFrame from the processed data
processed_df = pd.DataFrame({
    'title_entity_relation': title_entity_relations,
    'abstract_entity_relation': abstract_entity_relations
})


# Convert None values to empty dictionaries
processed_df['title_entity_relation'].fillna({}, inplace=True)
processed_df['abstract_entity_relation'].fillna({}, inplace=True)

# Convert DataFrame to JSON
processed_data_json = processed_df.to_json(orient='records', lines=True)

# Write JSON data to a file
with open('ER.json', 'w') as f:
    f.write(processed_data_json)
