# Individual Reflection — Alex Lim

I learned that retrieval determines the evidence available to the answer generator. The six paragraphs were stored as separate chunks, each retaining its text, ID, and filename. Keyword retrieval ranked chunks by shared distinct words after lowercasing and removing stopwords; embedding retrieval ranked them by cosine similarity to the question. I supplied the model with the question, selected passage text, and IDs, then checked whether each claim was supported by its cited passage.
One limitation was incomplete evidence coverage. For Q1, embeddings retrieved A2 and T1 but missed A1, whereas keywords retrieved both required passages. This showed me that semantically relevant passages can still omit a rule needed for a complete answer.
Next, I would test combining keyword and embedding retrieval, rerunning the same questions to check whether it recovers both A1 and A2 for Q1 and improves answer completeness without introducing unsupported claims.
