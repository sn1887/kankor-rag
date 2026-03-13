# Data guide

## Corpus principles

This assistant should be grounded in:

- officially distributable curriculum-aligned materials where licensing permits
- teacher-prep and explanation content aligned to exam subjects
- practice explanations and worked examples with explicit licensing
- metadata-rich corpus records that make retrieval auditable

It should not rely on scraped, unverifiable answer-key dumps as the primary knowledge base.

## Recommended metadata

- `title`
- `subject_category` - `math`, `natural_science`, `social_science`, `languages`
- `subject`
- `grade_band`
- `language` - `ps`, `fa`, `ar`, `en`
- `source_type` - `explanation`, `definition`, `worked_example`, `practice_question`
- `source_id`
- `page` - original textbook page number for exact citations
- `copyright`
- `license`
- `retrieval_weight`

## Versioning

Use explicit corpus versions such as `kankor-corpus@2025.09` and `kankor-corpus@2026.03` so answers can report the version used.

## Build command

```bash
python scripts/build_index.py   --input data/sample_corpus/kankor_sample.jsonl   --output-dir data/sample_index   --embedding-backend hash
```
