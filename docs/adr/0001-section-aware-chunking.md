# ADR 0001: Section-aware chunking over fixed windows

Status: accepted

## Context

Policies are written in sections: "Refund window", "Processing time", "Exceptions". A question is usually answered by one section, and a citation is only useful if it points a person to the right one. Fixed-size windows cut across headings, mix two rules in one chunk, and lose the heading that says what a sentence is about ("within 30 days" of what?).

## Options

1. Fixed windows of 40 words with 10 words of overlap (the common default).
2. One chunk per document.
3. One chunk per section, prefixed with the document title and heading, split on sentence boundaries with one sentence of overlap only when a section is long.

## Decision

Option 3. Each chunk reads like `Refund Policy > Refund window: Students may request...`, so the chunk carries its own context into both retrieval and the answer prompt, and a citation names a document and a section.

## Evidence

Measured on the 50 answerable golden questions (`rag eval`):

| Chunking | Embedder | Document Hit@1 | Section Hit@1 |
| --- | --- | --- | --- |
| fixed windows | hash | 0.90 | 0.38 |
| section-aware | hash | 0.90 | 0.68 |
| fixed windows | bge-small | 0.94 | 0.38 |
| section-aware | bge-small | 0.92 | 0.72 |

Finding the right document is about the same either way. Finding the right section, which is what the answer and the citation depend on, nearly doubles.

## Consequences

- Citations are human-checkable: "Annual Leave, Requests" rather than "chunk 17".
- Re-indexing is per document, so a policy edit re-embeds only that document's sections.
- Accepted cost: chunk sizes vary with the author's sections. Very long sections fall back to sentence windows; very short ones can lack context, which the title and heading prefix mitigates.
