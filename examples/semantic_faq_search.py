"""Answer customer questions by semantic search over an FAQ, no keyword overlap needed.

Use case: map free-text questions to the closest FAQ entry using sentence embeddings and
cosine similarity, with a threshold to hand off to a human when nothing matches well.

Needs: network on the first run to download sentence-transformers/all-MiniLM-L6-v2
(about 90 MB). Later runs work offline (set HF_HUB_OFFLINE=1).

Run: python examples/semantic_faq_search.py
"""

from __future__ import annotations

import numpy as np

from lingo import Pipeline

faq = {
    "How do I reset my password?": "Use 'Forgot password' on the sign-in page.",
    "What is your refund policy?": "Refunds are available within 30 days of delivery.",
    "How long does shipping take?": "Standard shipping takes 3-5 business days.",
    "Can I change my delivery address?": "Yes, until the order has been dispatched.",
}

embedder = Pipeline(task="embedding", model="sentence-transformers/all-MiniLM-L6-v2")
faq_questions = list(faq)
faq_vectors = np.array(embedder(faq_questions))
faq_vectors /= np.linalg.norm(faq_vectors, axis=1, keepdims=True)

THRESHOLD = 0.5
for question in [
    "I forgot my login credentials",
    "Can I get my money back?",
    "When will my parcel arrive?",
    "Do you sell gift cards?",
]:
    vector = np.array(embedder(question))
    scores = faq_vectors @ (vector / np.linalg.norm(vector))
    best = int(np.argmax(scores))
    if scores[best] >= THRESHOLD:
        print(
            f"{question!r}\n  -> {faq_questions[best]} ({scores[best]:.2f}): {faq[faq_questions[best]]}"
        )
    else:
        print(f"{question!r}\n  -> no confident match ({scores[best]:.2f}), route to an agent")

# The model wrapper also exposes pairwise similarity directly
print(f"\nsimilarity: {embedder.model.similarity('refund', 'money back'):.2f}")
