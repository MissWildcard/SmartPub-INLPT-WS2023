# SmartPub-INLPT-WS2023

We plan on using the medical dataset, comprised of abstracts from articles on PubMed concerning “intelligence” from 2013 till 2023 [Link to PubMed](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle%2Fabstract%5D&filter=simsearch1.fha&filter=years.2014-2024&sort=date).

Current overall plan:
- use OpenSearch to store/manage the dataset
- perform Document Retrieval on the dataset given the query (use OpenSearch tools/API)
- use extractive answer generation on the retrieved documents

## Questions:

1. Data
  - **Dataset PubMed**: Abstracts of the articles published between the years 2013 to 2023 that contain the word “intelligence” in the abstract’s text 
  - Apply algorithm from [PubMedQA dataset](https://arxiv.org/abs/1909.06146#:~:text=The%20task%20of%20PubMedQA%20is,k%20artificially%20generated%20QA%20instances) to create synthetic QA data for our dataset
  - what data do we store? (Metadata, Formatting)

2. Model
  - Fine-tune **T0** (**T5** pretrained on different general QA datasets)

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


----------------------------------------------------Minutes of the meeting--------------------------------------------------------------------------------------
----------------------------------------------------with John Ziegler--------------------------------------------------------------------------------------

Dec-18-2023

(S)Is there a minimum size for the dataset? Both dataset not very large, so use all of them.
If using the entire dataset makes the system slow, use a smaller dataset.
Extract keywords, other entities – should be incorporated in the model that you use.
Now focus on data – once its ready focus on model.
Split task among yourself – data extraction, the datamodel, setting up opensearch, ui etc
(J)Any idea on datamodel?  T5, T0
Metadata – keyword extraction, entities etc? 
Really imp task – preprocessing, storing data in doc model (there must be one workinh on that). Use bm25 as statistical model for 
Start simple and then go more advancedPM25 retreival should come first
If you start complex and your system not working – it’ll be difficult to debug the issue. So start simple and then go complex.
