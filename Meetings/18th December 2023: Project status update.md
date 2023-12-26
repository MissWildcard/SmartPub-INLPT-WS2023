# 1 Participants
- All 4 members in team
- Mentor - John Ziegler

# 2 Status

- Raw data (abstract text contains "intelligence") first set from PubMed extracted: FINISHED
- Several Data analyses: FINISHED
- Methodology: DECIDED
	+ Use method from **PubMedQA** as anchor to create set of training and test question-anwer pairs
	+ Fine-tune on **T0** model (T5 model that was finetuned on variety of general-domain QA datasets)

# 3 Mentor Advice

- Start with a more simple methodology first
- All data from PubMED should be extracted in full. If using the entire dataset makes the system slow, use a smaller dataset.
- Extract keywords, other entities – should be incorporated in the model that you use.
- Now focus on data – once its ready focus on model.
- Split task among yourself – data extraction, the datamodel, setting up opensearch, ui etc
- Metadata – keyword extraction, entities etc? 
- Really imp task – preprocessing, storing data in doc model (there must be one workinh on that). Use bm25 as statistical model for 
- Start simple and then go more advancedPM25 retreival should come first
- If you start complex and your system not working – it’ll be difficult to debug the issue. So start simple and then go complex.


# 4 Future approach
- Start an another methodology alongside with the planned methodology: Use Information Retrieval Method (**BM25**) to extract information from raw data for questions, and use a pre-trained LLM as embedding to create answers (*Planned*: **RoBERTa**)
- Working on other components (front-end)