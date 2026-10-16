"""RAGAS evaluation of the Arabic Legal RAG system, with MLflow logging and a CI gate.

Typical use
-----------
CI gate (20 questions, faithfulness only)::

    python -m arabic_legal_rag.evaluate_ragas --num-questions 20 --metrics faithfulness

Full run (>= 50 questions, all 4 metrics), gentle on a laptop::

    python -m arabic_legal_rag.evaluate_ragas --num-questions 50 --metrics all \
        --batch-size 5 --cooldown 20

Batching and resuming
---------------------
Scoring runs in batches (--batch-size, default 10). After EVERY batch:
  * the per-question scores are saved to outputs/ragas_scores.json, and
  * the running metric means are logged to MLflow (step = batch number).
If the run is stopped (Ctrl+C, overheating, rate limit, crash), nothing already scored
is lost: re-run the same command and only the unscored questions are judged. The MLflow
run of an interrupted execution is tagged evaluation_status=INCOMPLETE.
Use --no-cache to throw both caches away and start from scratch.

Configuration (environment variables)
-------------------------------------
Generator (the system under test; same variables as the API):
    GENERATOR_BACKEND  transformers | openai | extractive   (default: config.yaml)
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY                    (for backend=openai)

Judge (any OpenAI-compatible API; defaults to Groq's free tier):
    JUDGE_BASE_URL   default https://api.groq.com/openai/v1
    JUDGE_MODEL      default openai/gpt-oss-120b
    JUDGE_API_KEY    falls back to GROQ_API_KEY
    JUDGE_EXTRA_BODY optional JSON merged into the request, e.g. '{"reasoning_format":"hidden"}'
    JUDGE_MAX_TOKENS cap on judge output tokens per request (default 800). Keep it below your
                     provider's output-tokens-per-minute (OTPM) limit, or every request is
                     rejected with HTTP 429 no matter how long you wait.
    JUDGE_MAX_WORKERS parallel judge requests (default 1; raise only with a higher quota)

Generated answers are cached in outputs/ragas_answers.json so a crash or a
rate-limit stop never loses work. The cache key includes the retrieved context,
so changing retrieval automatically invalidates stale answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from loguru import logger

from arabic_legal_rag import tracing
from arabic_legal_rag.generator import build_generator
from arabic_legal_rag.rag import NOT_FOUND_AR, is_arabic, prepare_context
from arabic_legal_rag.utils import load_clean_corpus, load_config

DEFAULT_JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
MIN_VALID_RATIO = 0.9
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ----------------------------------------------------------------- helpers
def load_questions(path: str, n: int) -> list[dict]:
    """Load the first ``n`` evaluation questions; fail clearly if too few."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) < n:
        raise ValueError(
            f"{path} has {len(data)} questions but {n} were requested. "
            "Add more questions or lower --num-questions."
        )
    return data[:n]


def build_reference(question: dict, article_texts: dict[str, dict]) -> str:
    """Ground-truth reference = text of the expected articles, in the question's language."""
    field = "text_ar" if is_arabic(question["query"]) else "text_en"
    parts = []
    for art in question.get("expected_articles", []):
        record = article_texts.get(str(art))
        if record and record.get(field):
            parts.append(record[field])
    return "\n".join(parts)


def _cache_key(generator, question: str, items: list[dict]) -> str:
    blob = json.dumps(
        [generator.name, generator.model_name, question, items], ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic: a crash never leaves a half-written cache


MAX_RETRY_WAIT = 90  # seconds; a longer Retry-After means a daily quota, not a per-minute one


def generate_with_retry(generator, question, items, arabic, attempts: int = 8) -> str:
    """Call the generator, backing off on rate limits (HTTP 429) and 5xx errors.

    Honors the provider's Retry-After header and logs the provider's explanation, so a
    per-minute limit (wait and retry) can be told apart from a daily quota (fail fast).
    """
    import httpx

    for attempt in range(1, attempts + 1):
        try:
            return generator.generate(question, items, arabic)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            reason = e.response.text[:300].replace("\n", " ")
            if code not in (429, 500, 502, 503) or attempt == attempts:
                logger.error(f"Generator HTTP {code}: {reason}")
                raise

            wait = min(60, 5 * attempt)
            retry_after = e.response.headers.get("retry-after")
            if retry_after:
                try:
                    wait = float(retry_after) + 1
                except ValueError:
                    pass
            if wait > MAX_RETRY_WAIT:
                logger.error(
                    f"Generator HTTP {code} asks to wait {wait:.0f}s (likely a daily quota): "
                    f"{reason}"
                )
                raise
            logger.warning(
                f"Generator HTTP {code}; retrying in {wait:.0f}s ({attempt}/{attempts}): {reason}"
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def collect_rows(
    questions,
    retriever,
    generator,
    article_texts,
    cache_path: Path,
    use_cache=True,
    trace_ids: list | None = None,
    run_name: str | None = None,
):
    """Retrieve + generate an answer for each question; returns RAGAS-style rows.

    If Langfuse is configured, every question also gets a trace (ask -> retrieve + generate)
    and its trace id is appended to ``trace_ids`` (same order as the questions), so the
    RAGAS scores can be attached to the right trace afterwards. ``None`` entries mean
    tracing is off.
    """
    cache = _load_cache(cache_path) if use_cache else {}
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    for i, q in enumerate(questions, 1):
        question = q["query"]
        with tracing.span(
            "ask",
            tags=["ragas-eval"] + ([run_name] if run_name else []),
            input={"question": question},
            metadata={"endpoint": "ragas-eval", "expected_articles": q.get("expected_articles")},
        ) as root:
            with tracing.span("retrieve", input={"question": question}) as r:
                items, sources, arabic = prepare_context(retriever, question)
                tracing.safe_update(r, output={"sources": sources, "documents": len(items)})
            key = _cache_key(generator, question, items)

            with tracing.span(
                "generate",
                as_type="generation",
                model=generator.model_name,
                input={"question": question, "contexts": items},
                metadata={"backend": generator.name, "cached": key in cache},
            ) as g:
                if key in cache:
                    answer = cache[key]
                    logger.info(f"[{i}/{len(questions)}] cached answer")
                else:
                    logger.info(f"[{i}/{len(questions)}] generating answer")
                    answer = (
                        generate_with_retry(generator, question, items, arabic)
                        if items
                        else NOT_FOUND_AR
                    )
                    cache[key] = answer
                    _save_cache(cache_path, cache)
                tracing.safe_update(g, output=answer)

            tracing.safe_update(root, output={"answer": answer, "sources": sources})
            if trace_ids is not None:
                trace_ids.append(tracing.trace_id_of(root))

        rows["user_input"].append(question)
        rows["response"].append(answer)
        rows["retrieved_contexts"].append(
            [f"[{it['article']}]: {it['text']}" for it in items] or [""]
        )
        rows["reference"].append(build_reference(q, article_texts))
    return rows


# ------------------------------------------------------------------- RAGAS
def run_ragas(rows: dict, metric_choice: str, embeddings):
    """Score the rows with RAGAS using the configured judge; returns a DataFrame."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set JUDGE_API_KEY (or GROQ_API_KEY) to use the judge model.")

    judge = ChatOpenAI(
        model=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        base_url=os.getenv("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
        api_key=api_key,
        temperature=0,
        max_retries=6,
        max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "800")),
        extra_body=json.loads(os.getenv("JUDGE_EXTRA_BODY") or "{}") or None,
    )
    eval_llm = LangchainLLMWrapper(judge)

    if metric_choice == "faithfulness":
        metrics = [faithfulness]
    else:
        answer_relevancy.strictness = 1  # many hosted APIs reject n>1 completions
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        if not all(rows["reference"]):
            raise ValueError("All 4 metrics need a reference for every question.")

    dataset = Dataset.from_dict(rows)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(
            max_workers=int(os.getenv("JUDGE_MAX_WORKERS", "1")), timeout=240, max_retries=10
        ),
    )
    return result.to_pandas()


# --------------------------------------------------- batched, resumable scoring
def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _metrics_for(metric_choice: str) -> list[str]:
    return ["faithfulness"] if metric_choice == "faithfulness" else list(METRIC_NAMES)


def _score_key(rows: dict, i: int, metric_choice: str, judge_model: str) -> str:
    """Cache key for one question's scores: changes if the answer, context or judge changes."""
    blob = json.dumps(
        [
            judge_model,
            metric_choice,
            rows["user_input"][i],
            rows["response"][i],
            rows["retrieved_contexts"][i],
            rows["reference"][i],
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def mean_scores(per_question: list[dict], names: list[str]) -> dict:
    """Mean of each metric over the questions that have a valid (non-missing) score."""
    means = {}
    for name in names:
        vals = [s[name] for s in per_question if not _is_missing(s.get(name))]
        means[name] = sum(vals) / len(vals) if vals else None
    return means


def score_in_batches(
    rows: dict,
    metric_choice: str,
    judge_model: str,
    embeddings,
    scores_path: Path,
    batch_size: int = 10,
    cooldown: float = 0.0,
    use_cache: bool = True,
    on_batch=None,
    scorer=None,
) -> list[dict]:
    """Score questions batch by batch, saving after each batch so a stop loses nothing.

    Returns one ``{metric: value-or-None}`` dict per question, in question order.
    ``on_batch(batch_no, n_batches, scored_so_far, total, per_question_so_far)`` is called
    after every batch (used for MLflow logging).
    """
    scorer = scorer or run_ragas
    names = _metrics_for(metric_choice)
    n = len(rows["user_input"])
    cache = _load_cache(scores_path) if use_cache else {}
    keys = [_score_key(rows, i, metric_choice, judge_model) for i in range(n)]

    todo = [i for i, k in enumerate(keys) if k not in cache]
    logger.info(f"{n - len(todo)}/{n} questions already scored (cache); {len(todo)} to score")
    batches = [todo[s : s + batch_size] for s in range(0, len(todo), batch_size)]

    for b, idxs in enumerate(batches, 1):
        sub = {col: [vals[i] for i in idxs] for col, vals in rows.items()}
        logger.info(f"Scoring batch {b}/{len(batches)} ({len(idxs)} questions)")
        df = scorer(sub, metric_choice, embeddings)

        failed = 0
        for pos, i in enumerate(idxs):
            entry = {}
            for m in names:
                value = df[m].iloc[pos] if m in df.columns else None
                entry[m] = None if _is_missing(value) else float(value)
            if all(v is None for v in entry.values()):
                # Every judgment failed (typically a 429 rate limit or daily quota).
                # Do NOT cache it, so the next run retries this question.
                failed += 1
                continue
            cache[keys[i]] = entry
        _save_cache(scores_path, cache)  # progress is on disk before anything else happens

        if failed == len(idxs) and len(idxs) >= 3:  # tiny batches can be NaN legitimately
            logger.error(
                f"All {failed} judgments in batch {b} failed (likely the judge's rate limit "
                "or daily quota). Stopping; scores so far are saved. Wait, then re-run the "
                "same command to resume."
            )
            break
        if failed:
            logger.warning(f"{failed} question(s) got no score and will be retried next run")

        if on_batch is not None:
            done = [cache[k] for k in keys if k in cache]
            on_batch(b, len(batches), len(done), n, done)
        if cooldown and b < len(batches):
            logger.info(f"Cooling down for {cooldown:.0f}s ...")
            time.sleep(cooldown)

    return [cache.get(k, dict.fromkeys(names)) for k in keys]


# -------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--num-questions", type=int, default=20)
    p.add_argument("--metrics", choices=["faithfulness", "all"], default="faithfulness")
    p.add_argument("--threshold", type=float, default=0.75, help="faithfulness CI gate")
    p.add_argument("--dataset", default="data/eval_dataset.json")
    p.add_argument("--answers-cache", default="outputs/ragas_answers.json")
    p.add_argument("--scores-cache", default="outputs/ragas_scores.json")
    p.add_argument("--batch-size", type=int, default=10, help="questions judged per batch")
    p.add_argument("--cooldown", type=float, default=0.0, help="seconds to pause between batches")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--report-dir", default="reports")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config("configs/config.yaml")

    questions = load_questions(args.dataset, args.num_questions)  # fail fast, before models load
    corpus = load_clean_corpus(config["data"]["json_path"])
    article_texts = {str(r["article_number"]): r for r in corpus}

    import mlflow
    import pandas as pd

    from arabic_legal_rag.retriever import get_legal_retriever

    index_path = config["vector_db"]["index_dir"]
    top_k = int(config.get("model", {}).get("top_k", 3))
    retriever = get_legal_retriever(index_path=index_path, k=top_k)
    generator = build_generator(config.get("generation", {}))
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    names = _metrics_for(args.metrics)

    trace_ids: list = []  # Langfuse trace id per question (None when tracing is off)
    rows = collect_rows(
        questions, retriever, generator, article_texts,
        Path(args.answers_cache), use_cache=not args.no_cache,
        trace_ids=trace_ids, run_name=f"ragas_{args.metrics}_{len(questions)}q",
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(config["mlflow"]["experiment_name"] + "_ragas")

    with mlflow.start_run(run_name=f"ragas_{args.metrics}_{len(questions)}q"):
        mlflow.log_params({
            "generator_backend": generator.name, "generator_model": generator.model_name,
            "judge_model": judge_model, "embedding_model": config["model"]["embedding_model"],
            "top_k": top_k, "num_questions": len(questions), "metrics": args.metrics,
            "threshold": args.threshold, "batch_size": args.batch_size,
        })
        mlflow.set_tag("evaluation_status", "RUNNING")

        def log_batch(batch_no, n_batches, scored, total, per_question):
            running = mean_scores(per_question, names)
            for name, value in running.items():
                if value is not None:
                    mlflow.log_metric(f"ragas_{name}_running", value, step=batch_no)
            mlflow.log_metric("questions_scored", scored, step=batch_no)
            shown = running.get("faithfulness")
            shown = "n/a" if shown is None else f"{shown:.3f}"
            logger.info(f"Batch {batch_no}/{n_batches} done: {scored}/{total} scored, "
                        f"running faithfulness {shown}")

        logger.info(f"Running RAGAS ({args.metrics}) with judge {judge_model} "
                    f"in batches of {args.batch_size} ...")
        try:
            per_question = score_in_batches(
                rows, args.metrics, judge_model, retriever.vectorstore.embedding_function,
                Path(args.scores_cache), batch_size=args.batch_size, cooldown=args.cooldown,
                use_cache=not args.no_cache, on_batch=log_batch,
            )
        except BaseException:
            mlflow.set_tag("evaluation_status", "INCOMPLETE")
            logger.error("Scoring stopped early. Scores so far are saved; "
                         "re-run the same command to resume.")
            raise

        # Attach every RAGAS score to the Langfuse trace of the question it judged.
        for trace_id, scores in zip(trace_ids, per_question):
            for name in names:
                tracing.score_trace(trace_id, name, scores.get(name),
                                    comment=f"RAGAS, judge={judge_model}")
        tracing.flush()

        df = pd.DataFrame(rows)
        if any(trace_ids):
            df["langfuse_trace_id"] = trace_ids
        for name in names:
            df[name] = pd.to_numeric(pd.Series([s.get(name) for s in per_question]),
                                     errors="coerce")

        summary = {"num_questions": len(df), "judge_model": judge_model,
                   "generator": f"{generator.name}:{generator.model_name}"}
        for name in METRIC_NAMES:
            if name in df.columns:
                summary[name] = None if df[name].notna().sum() == 0 else float(df[name].mean())
                summary[f"{name}_valid"] = int(df[name].notna().sum())

        faith = summary.get("faithfulness")
        valid_ratio = summary.get("faithfulness_valid", 0) / max(len(df), 1)
        if valid_ratio < MIN_VALID_RATIO:
            status = "INVALID"
        elif faith is not None and faith >= args.threshold:
            status = "PASSED"
        else:
            status = "FAILED"
        summary["status"] = status

        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        results_csv = report_dir / "ragas_results.csv"
        summary_json = report_dir / "ragas_summary.json"
        df.to_csv(results_csv, index=False, encoding="utf-8-sig")
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                encoding="utf-8")

        for name in METRIC_NAMES:
            if summary.get(name) is not None:
                mlflow.log_metric(f"ragas_{name}", summary[name])
        mlflow.log_metric("valid_ratio", valid_ratio)
        mlflow.set_tag("evaluation_status", status)
        mlflow.log_artifact(str(results_csv))
        mlflow.log_artifact(str(summary_json))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if status == "PASSED":
        print(f"✅ PASSED: faithfulness {faith:.3f} >= {args.threshold}")
    elif status == "INVALID":
        print(f"❌ INVALID: only {valid_ratio:.0%} of judgments were valid (need {MIN_VALID_RATIO:.0%}).")
    else:
        shown = "n/a" if faith is None else f"{faith:.3f}"
        print(f"❌ FAILED: faithfulness {shown} < {args.threshold} (CI gate failed)")
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())