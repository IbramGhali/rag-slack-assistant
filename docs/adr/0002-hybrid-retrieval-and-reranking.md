# ADR 0002: Retrieval method, decided by the measured table

Status: accepted

## Context

Keyword search (BM25) is strong on exact terms such as "$25", "extension 4400" or a policy name, and weak on paraphrase ("dodgy link" for "phishing"). Dense embeddings handle paraphrase and miss exact tokens they have not seen. The choice should be made on data, not habit.

## Options

BM25, dense, hybrid (both rankings fused with reciprocal rank fusion), each with and without a cross-encoder re-ranker over the top 20 candidates.

## Evidence

50 answerable golden questions, section-aware chunks, `bge-small` embeddings (`make semantic`):

| Method | Hit@1 | MRR | Section Hit@1 | Paraphrase Hit@1 |
| --- | --- | --- | --- | --- |
| BM25 | 0.90 | 0.93 | 0.68 | 0.75 |
| Dense | 0.96 | 0.98 | 0.92 | 0.90 |
| Hybrid (RRF) | 0.92 | 0.96 | 0.72 | 0.80 |
| Dense + re-rank | 0.98 | 0.99 | 0.96 | 0.95 |
| Hybrid + re-rank | 0.98 | 0.99 | 0.96 | 0.95 |

Two findings. Without re-ranking, plain RRF hybrid scored below dense alone on this corpus, because BM25 promotes sections that share words with the question but answer a different point ("Partial refunds" also mentions the "refund window"). With a cross-encoder re-ranker, dense and hybrid tie at the top.

## Decision

The recommended configuration is hybrid retrieval with the cross-encoder re-ranker. It equals dense + re-rank on this golden set, and keeps BM25 candidates in the pool for exact identifiers (extension numbers, amounts, document names), which embeddings handle worst and this golden set does not yet test. The offline default is hybrid on the hash embedder, which needs no model download.

## Consequences

- The re-ranker is the expensive step. Measured on a laptop CPU over the 60 golden questions: p50 12 ms per question with bge-small alone, 974 ms with the re-ranker scoring 20 candidates. A one-second reply is acceptable in Slack, where the answer arrives as a threaded message, but it is the first thing to tune: fewer candidates, a GPU, or a hosted re-ranker.
- The decision is revisited by re-running the table. A golden-set slice of exact-identifier questions is on the roadmap so the BM25 argument is measured, not asserted.
