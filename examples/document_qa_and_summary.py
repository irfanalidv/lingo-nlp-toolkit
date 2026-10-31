"""Summarise a policy document and answer questions about it with exact source spans.

Use case: a long internal document where readers want a short summary plus answers that
point to the exact sentence they came from (character offsets into the original text).

Needs: network on the first run to download deepset/roberta-base-squad2 (about 500 MB) and
facebook/bart-large-cnn (about 1.6 GB). Later runs work offline (set HF_HUB_OFFLINE=1).

Run: python examples/document_qa_and_summary.py
"""

from __future__ import annotations

from lingo import Pipeline

policy = (
    "Employees may work remotely up to three days per week, subject to manager approval. "
    "Requests must be submitted through the HR portal at least two weeks in advance. "
    "Remote workers are expected to be reachable between 10:00 and 16:00 local time. "
    "The company reimburses up to 40 euros per month for home internet costs, provided "
    "receipts are uploaded before the end of the following month. Equipment such as "
    "laptops and monitors remains company property and must be returned when employment "
    "ends. The policy is reviewed every January by the People Operations team."
)

qa = Pipeline(task="question-answering", model="deepset/roberta-base-squad2")
for question in [
    "How many days per week can employees work remotely?",
    "How much is reimbursed for internet?",
    "Who reviews the policy?",
]:
    answer = qa(question=question, context=policy)
    span = policy[answer["start"] : answer["end"]]
    print(
        f"Q: {question}\nA: {answer['answer']}  (score {answer['score']:.2f}, chars {answer['start']}-{answer['end']}: {span!r})"
    )

summarizer = Pipeline(task="summarization", model="facebook/bart-large-cnn")
summary = summarizer(policy, max_length=60, min_length=20, num_beams=4)
print(f"\nSummary: {summary['summary_text']}")
