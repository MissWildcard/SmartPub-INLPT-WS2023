# SmartPub-INLPT-WS2023

We plan on using the medical dataset, comprised of abstracts from articles on PubMed concerning “intelligence” from 2013 till 2023 [Link to PubMed](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle%2Fabstract%5D&filter=simsearch1.fha&filter=years.2014-2024&sort=date).

Current overall plan:
- use OpenSearch to store/manage the dataset
- perform Document Retrieval on the dataset given the query (use OpenSearch tools/API)
- use extractive answer generation on the retrieved documents

## Questions:

1. Data
  - what kind of embeddings do we want to use?
  - do we want to store embedded or text documents?
  - what data do we store? (Metadata, Formatting)

2. Model
  - are there requirements concerning extractive/generative answer generation?
  - what kind of pre-trained models can we use?

## Related Works
Our research to related works and general ideas can be found in this [Google Doc](https://docs.google.com/document/d/1m4kF7XVmnPf96O8Rb5n4UKLFJVD0xe2SMmiUjc5mVgk/edit?pli=1#heading=h.vpnyzzywsw32).

## Possible Implementation

- OpenSearch Environment
  - Docker
  - VS-Code

- Document Retrieval
  - keywords (tf-idf)
  - k-Nearest Neighbours

## Long term goals:
- add context/'proof' to answers
- nice UI
