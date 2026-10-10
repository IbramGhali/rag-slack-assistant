# ADR 0003: Refuse below an evidence threshold rather than answer weakly

Status: accepted

## Context

When nothing in the documents answers a question, retrieval still returns the "closest" chunks, and a language model given loosely related text will often produce a confident, plausible and wrong answer. For a policy assistant, a wrong answer about leave or refunds is worse than "I can't find that".

## Options

1. Always answer; rely on the prompt to make the model admit ignorance.
2. Refuse when the best chunk's similarity to the question is below a threshold, before calling the model; also let the model refuse, and refuse when no citation survives checking.

## Decision

Option 2, with three layers:

1. **Evidence gate in code.** If the best retrieved chunk's cosine similarity is below the threshold, the assistant refuses without calling the model.
2. **Model refusal.** The prompt asks the model to refuse when the sources do not answer the question.
3. **Citation check.** Citations must point at retrieved sources. If none survive, the answer is replaced with the refusal.

The threshold is chosen from a sweep over the golden set (`rag eval --sweep`), separately for each embedder, because similarity scales differ:

| bge-small threshold | Unanswerable refused | Answerable still answered |
| --- | --- | --- |
| 0.60 | 30% | 98% |
| **0.65** | **90%** | **96%** |
| 0.70 | 100% | 90% |

0.65 is the default for bge-small (0.25 for the offline hash embedder). With the model-refusal layer on top, all 10 unanswerable questions are refused, with 3 of 50 answerable questions wrongly refused.

## Consequences

- Refusals are cheap and safe: no model call, no hallucination.
- The threshold is tuned on the same 60 questions it is reported on, which flatters it. A held-out set and per-area thresholds are on the roadmap.
- Accepted cost: a few answerable but oddly phrased questions are refused. The refusal message tells the person to rephrase or ask the policy owner, and the "Not helpful" button records it for review.
