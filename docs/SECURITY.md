# Security, safety, and governance

## Product boundaries

This project is an exam-prep assistant, not an official score authority and not an answer-key oracle.

The system should:

- prefer grounded explanations over unsupported certainty
- indicate when retrieval evidence is weak
- expose the source snippets that informed the answer
- track corpus version for auditability

## Prompt injection handling

Retrieved text is untrusted input. The current design passes retrieved passages as quoted context instead of system instructions and keeps policy in the system prompt.

## Operational guidance

- avoid storing raw user chat logs by default on the pilot
- keep tokens server-side only
- verify downloaded artifact checksums in production
- cap prompt and response lengths on free-tier infra
