# ADR 0004: Superseded policies are demoted, not deleted

Status: accepted

## Context

Policies change. The old version stays relevant for questions about what applied at the time, and for audit, but an assistant that answers "20 days of annual leave" after the policy moved to 22 is wrong in a way that costs money and trust.

## Options

1. Delete superseded documents from the index.
2. Keep them with no special handling.
3. Keep them, mark them with `superseded_by` metadata, multiply their retrieval score by a penalty (0.5 by default) and subtract a margin after re-ranking, and label them in the Slack answer if they are ever cited.

## Decision

Option 3. Each document's front matter carries `effective` and `superseded_by`. Retrieval demotes superseded chunks so the current version wins when both match. They remain searchable, so a question that names the old version still finds it, and any citation of an old version is labelled "(superseded version)" in Slack.

## Consequences

- On the 10 golden questions whose answer changed between versions, the current policy is retrieved first for all 10 with either embedder. Answers come from the current policy in 10 of 10 with bge-small (9 of 10 offline), and no answer contains the old value.
- History is preserved for audit and "what used to apply" questions.
- Accepted cost: a superseded policy that should have been retired entirely still costs index space. Retirement is a separate, explicit deletion in the corpus.
