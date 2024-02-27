# 1 Participants
- All 4 members in team
- Mentor - John Ziegler

# 2 Status

- Report on changing from using FINETUNED T0 into using RAG and paper work KAPING framework (Baek et al. 2023)


# 3 Current Approach
- Back-End
	+ Turn all texts in article titles and article abstracts into sequences of tuples (KG) in form of (entity1, relation, entity2) and saved in the database
	+ Extract the entities from the question
	+ Use those entities to look for appropriate relations from the database
	+ Find those that are most similar to the sentence embedding of the question (top 10) and use them as the context
	+ Feed these with the question and the context into a QA-pre-trained model to retrieve the answer

- Front-End
	+ Build a simple front-end web-base to support the Back-End