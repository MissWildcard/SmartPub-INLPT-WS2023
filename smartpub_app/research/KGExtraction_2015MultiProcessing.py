'''An attempt to KG extraction with multiprocessing'''

import pandas as pd
from transformers import pipeline, AutoTokenizer
import nltk
from nltk.tokenize import word_tokenize
import multiprocessing


#CHANGE THE YEAR TO THE SPECIFIC YEAR YOU NEED
year = 2015

# Initialize the triplet extraction pipeline using Babelscape/rebel-large model
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

# Load the tokenizer for the desired model
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
max_token_length = tokenizer.model_max_length

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

def processYear(year):
    # Load the CSV file into a DataFrame
    df_all = pd.read_csv("AllData.csv")
    df = df_all[df_all['Year'] == year]
    print("Count of rows in df:", len(df))

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

    # Write JSON data to output file
    with open(f'ER_{year}.json', 'w') as f:
        f.write(processed_data_json)

def genAbstractTextFromPipeline(abs):
    # If abstract length exceeds the maximum token length, use sliding window approach
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
            triplets = extract_triplets(generated_abs_text_decoded)
            # Append the triplets to the list
            triplets_list.extend(triplets)
        return triplets_list
    
    else:
        # Generate abstract text using the triplet extraction pipeline
        generated_abs_text = triplet_extractor(abs, return_tensors=True, return_text=False)
        generated_abs_text_decoded = triplet_extractor.tokenizer.batch_decode([generated_abs_text[0]["generated_token_ids"]])[0]
        # Extract and return triplets from the generated text
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

'''
def process_data_for_year(year):
    # Load the CSV file into a DataFrame
    df_all = pd.read_csv("AllData.csv")
    df = df_all[df_all['Year'] == year] 
    print("Count of rows in df:", len(df))

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

    # Write JSON data to output file
    with open(f'ER_{year}.json', 'w') as f:
        f.write(processed_data_json)
'''

if __name__ == '__main__':
    # Start multiprocessing for the specified year
    processYear(year)


#%%
