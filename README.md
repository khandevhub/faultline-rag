# Incident Intelligence RAG

A Retrieval-Augmented Generation system built to explore what actually matters in real RAG pipelines: retrieval quality, failure modes, and safe answers.

This project is not a chatbot demo. It’s an experiment in how large language models behave when their outputs are strictly bounded by retrieved evidence — and what happens when that evidence is weak or misleading.

The system answers when it should, and refuses when it shouldn’t.

----------------------------------

How it works, end to end:

Incident data is normalized into a consistent schema and split into semantically meaningful sections such as summaries, root causes, resolutions, and lessons learned. These sections are embedded using a retrieval-optimized model and indexed with FAISS for similarity search.

When a query comes in, the system does not rely on semantic similarity alone.

Semantic scores are combined with lexical overlap signals to form a hybrid retrieval score. This helps avoid common failure cases where embeddings over-generalize and retrieve “kind of related” but ultimately wrong context.

Low-confidence matches are filtered out. If nothing passes the threshold, the system refuses to answer instead of hallucinating.

If relevant context exists, it is assembled into a constrained prompt and passed to a lightweight instruction-tuned model for generation.

----------------------------------

The pipeline, conceptually:

Incident data  
→ normalization  
→ incident-aware chunking  
→ embeddings  
→ FAISS index  
→ hybrid retrieval  
→ thresholding  
→ grounded generation  

Each stage is implemented as a separate module so behavior can be inspected and changed without touching the rest of the system.

---------------------------

Why hybrid retrieval exists here:

Early versions of the system used semantic similarity only. Evaluation showed this caused false positives, especially for generic infrastructure queries and loosely related operational terms.

Adding lexical overlap as a secondary signal reduced irrelevant matches without hurting recall. This change was driven by measurement, not intuition.

Hybrid retrieval is now the default.

---

Refusal is a feature, not a bug.

If the system cannot find strong evidence in the indexed data, it explicitly refuses to answer. This prevents the model from filling gaps with prior knowledge or guesses.

Answering less often, but more correctly, is the goal.

---

Evaluation is part of the project, not an afterthought.

Retrieval quality is measured using simple but informative metrics:
precision at K, refusal correctness, and false positives.

These metrics directly informed threshold tuning and retrieval strategy changes.

On a small controlled dataset, the system achieves high precision while correctly refusing out-of-domain queries.

--------------------------

Model choice is intentional.

A small instruction-tuned model is used locally to keep behavior reproducible and to ensure that answers come from retrieved context, not memorized knowledge.

The rest of the system is model-agnostic and can be paired with larger open-source models or hosted APIs without refactoring.

-------------------------------

Where this breaks as data grows:

As the corpus expands, semantic similarity becomes noisier and retrieval needs stronger ranking and indexing strategies. Lexical signals become more important, and learned re-rankers start to make sense.

The current architecture is designed so these upgrades slot in cleanly.

-----------------------------

Tech stack:

Python, FAISS, SentenceTransformers, Hugging Face Transformers.

No paid APIs. No black boxes.

-----------------------------

This project exists to understand why RAG systems fail, not just how to make them work.

Correct answers matter. Refusing when unsure matters more.
