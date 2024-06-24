**RAGAS**

A synthetic dataset is made using the framework. The synthetic questions and answers are in three categories: 

1. ``Simple`` *What is the time frame for partial resumption of work after childbirth?*
- Answer: *Delvis genoptagelse af arbejdet kan ske i fraværet i de første 14 uger efter fødslen og i op til de 32 uger herefter.*

2. ``Reasoning`` *What is the condition for selecting a fællestillidsrepræsentant in relation to the local agreement and what is the recommended inclusion to ensure utilization of this right?*
- Answer: *The condition for selecting a fællestillidsrepræsentant in relation to the local agreement is that there should be a tilkendegivelse (indication) from the overenskomstgrupper (collective agreement groups) who wish to exercise this right. The recommended inclusion to ensure utilization of this right is to include a list of the personnel organizations/overenskomstgrupper who have expressed their desire to exercise this right in the local agreement on co-determination and participation*

3. ``Multi context`` *What is the role of the MED-aftale in relation to selvejende institutioner' obligations for APV and arbejdsmiljøuddannelse?*
- Answer: *The local MED-aftale can contain principles for the implementation of APV and mandatory arbejdsmiljøuddannelse that the selvejende institutioner in the joint arbejdsmiljøorganisation are obligated to follow*

**Metrics**

The four metrics used are: 

1. ``Answer relevancy`` The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the question, the context and the answer. The Answer Relevancy is defined as the mean cosine similarity of the original question to a number of artifical questions, which where generated (reverse engineered) based on the answer.
2. ``Context precision`` Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the question, ground_truth and the contexts, with values ranging between 0 and 1, where higher scores indicate better precision.
3. ``Context recall`` Context recall measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance. To estimate context recall from the ground truth answer, each sentence in the ground truth answer is analyzed to determine whether it can be attributed to the retrieved context or not.
4. ``Faithfulness`` This measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better. The generated answer is regarded as faithful if all the claims that are made in the answer can be inferred from the given context. To calculate this a set of claims from the generated answer is first identified. Then each one of these claims are cross checked with given context to determine if it can be inferred from given context or not

These can be read about here: [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)"# RAG_notes" 
