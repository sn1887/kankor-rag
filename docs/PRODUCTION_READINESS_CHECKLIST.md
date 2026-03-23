# RAG Production Readiness Checklist

Use this checklist after each retrieval/generation benchmark run.

## 1) Data + Retrieval

- Verify `retrieval_results.jsonl` exists and includes all planned queries.
- Ensure top-1 retrieval is not overly concentrated in one window.
- Ensure top-1 vs top-2 score margin is healthy (not near-zero for most queries).
- For labeled queries, verify expected page ranges appear in top-1/top-k.

## 2) Generation Quality

- Ensure `generation_results.jsonl` exists for all retrieval rows.
- Check truncated/incomplete outputs are rare.
- Check answers follow Dari-first policy (except required English fragments).
- Check greeting queries return greeting/help behavior instead of content hallucination.
- Check no-evidence responses are explicit when retrieval evidence is weak.

## 3) Cost Controls

- Keep `window-pages=2` for lower generation context cost.
- Keep `answer-top-k=1` as default budget mode.
- Increase only when measured quality gain justifies extra spend.

## 4) Operational Safety

- Log query text, top-k windows, score margins, and selected windows for generation.
- Track failure rate by query intent (`greeting`, `concept_explain`, `practice_questions`, etc.).
- Track answer language compliance and truncation rate over time.

## 5) Acceptance Gate

Run:

```bash
python scripts/rag_eval_report.py --experiment-dir <RUN_DIR>
```

Pass criteria are defined in `docs/RAG_EVAL_THRESHOLDS.json`.
