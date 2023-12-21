# 1 Participants
- All 4 members in team
- Mentor

# 2 Status

- Raw data (abstract text contains "intelligence") first set from PubMed extracted: FINISHED
- Several Data analyses: FINISHED
- Methodology: DECIDED
	+ Use method from **PubMedQA** as anchor to create set of training and test question-anwer pairs
	+ Fine-tune on **T0** model (T5 model that was finetuned on variety of general-domain QA datasets)

# 3 Mentor Advice

- Start with a more simple methodology first
- All data from PubMED should be extracted in full

# 4 Future approach
- Start an another methodology alongside with the planned methodology: Use Information Retrieval Method (**BM25**) to extract information from raw data for questions, and use a pre-trained LLM as embedding to create answers (*Planned*: **RoBERTa**)
- Working on other components (front-end)