"""Triage support tickets: sentiment plus the companies, people and places they mention.

Use case: a support queue where angry tickets that name a partner company or a city should
be routed first. Two pipelines share the same API: sentiment analysis and NER with entity
grouping.

Needs: network on the first run to download two models from the Hugging Face Hub
(cardiffnlp/twitter-roberta-base-sentiment-latest, about 500 MB, and dslim/bert-base-NER,
about 430 MB). Later runs work offline (set HF_HUB_OFFLINE=1).

Run: python examples/support_ticket_triage.py
"""

from __future__ import annotations

from lingo import Pipeline

tickets = [
    "My order from Acme Corp never arrived in Berlin and nobody answers my emails.",
    "Thanks Sarah, the replacement charger works perfectly!",
    "The app crashes every time I open the Paris store page. Very frustrating.",
    "Can I change the delivery address for order 1042?",
]

sentiment = Pipeline(
    task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
# Extra keyword arguments are forwarded to the Hugging Face pipeline. "first" groups
# sub-word pieces into whole words ("simple" can split a word such as "Acme" in two).
ner = Pipeline(task="ner", model="dslim/bert-base-NER", aggregation_strategy="first")

moods = sentiment.batch_predict(tickets, batch_size=8)
entities = ner(tickets)

rows = []
for text, mood, ents in zip(tickets, moods, entities, strict=True):
    names = [f"{e['word']} ({e['entity_group']})" for e in ents if e["score"] > 0.8]
    priority = "HIGH" if mood["label"] == "negative" and names else "normal"
    rows.append((priority, mood["label"], mood["score"], names, text))

for priority, label, score, names, text in sorted(rows, key=lambda r: r[0] != "HIGH"):
    print(f"[{priority:^6}] {label:<8} {score:.2f}  {', '.join(names) or '-':<32} {text[:55]}")
