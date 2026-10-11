"""RAGAS evaluation of the Arabic Legal RAG system, with MLflow logging and a CI gate.

Typical use
-----------
CI gate (20 questions, faithfulness only)::

    python -m arabic_legal_rag.evaluate_ragas --num-questions 20 --metrics faithfulness

Full run (>= 50 questions, all 4 metrics)::

    python -m arabic_legal_rag.evaluate_ragas --num-questions 50 --metrics all

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

Generated answers are cached in outputs/ragas_answers.json so a crash or a
rate-limit stop never loses work. The cache key includes the retrieved context,
so changing retrieval automatically invalidates stale answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

from arabic_legal_rag.generator import build_generator
from arabic_legal_rag.rag import NOT_FOUND_AR, is_arabic, prepare_context
from arabic_legal_rag.utils import load_clean_corpus, load_config

DEFAULT_JUDGE_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"
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


def generate_with_retry(generator, question, items, arabic, attempts: int = 5) -> str:
    """Call the generator, backing off on rate limits (HTTP 429) and 5xx errors."""
    import httpx

    for attempt in range(1, attempts + 1):
        try:
            return generator.generate(question, items, arabic)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code not in (429, 500, 502, 503) or attempt == attempts:
                raise
            wait = min(60, 5 * attempt)
            logger.warning(f"Generator HTTP {code}; retrying in {wait}s ({attempt}/{attempts})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def collect_rows(questions, retriever, generator, article_texts, cache_path: Path, use_cache=True):
    """Retrieve + generate an answer for each question; returns RAGAS-style rows."""
    cache = _load_cache(cache_path) if use_cache else {}
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    for i, q in enumerate(questions, 1):
        question = q["query"]
        items, _sources, arabic = prepare_context(retriever, question)
        key = _cache_key(generator, question, items)

        if key in cache:
            answer = cache[key]
            logger.info(f"[{i}/{len(questions)}] cached answer")
        else:
            logger.info(f"[{i}/{len(questions)}] generating answer")
            answer = (
                generate_with_retry(generator, question, items, arabic) if items else NOT_FOUND_AR
            )
            cache[key] = answer
            _save_cache(cache_path, cache)

        rows["user_input"].append(question)
        rows["response"].append(answer)
        rows["retrieved_contexts"].append([it["text"] for it in items] or [""])
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
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
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
        run_config=RunConfig(max_workers=2, timeout=240, max_retries=6),
    )
    return result.to_pandas()


# -------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--num-questions", type=int, default=20)
    p.add_argument("--metrics", choices=["faithfulness", "all"], default="faithfulness")
    p.add_argument("--threshold", type=float, default=0.75, help="faithfulness CI gate")
    p.add_argument("--dataset", default="data/eval_dataset.json")
    p.add_argument("--answers-cache", default="outputs/ragas_answers.json")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config("configs/config.yaml")

    questions = load_questions(args.dataset, args.num_questions)  # fail fast, before models load
    corpus = load_clean_corpus(config["data"]["json_path"])
    article_texts = {str(r["article_number"]): r for r in corpus}

    import mlflow
    import pandas as pd  # noqa: F401  (ensures pandas is present for to_pandas)

    from arabic_legal_rag.retriever import get_legal_retriever

    index_path = config["vector_db"]["index_dir"]
    top_k = int(config.get("model", {}).get("top_k", 3))
    retriever = get_legal_retriever(index_path=index_path, k=top_k)
    generator = build_generator(config.get("generation", {}))
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    rows = collect_rows(
        questions, retriever, generator, article_texts,
        Path(args.answers_cache), use_cache=not args.no_cache,
    )

    logger.info(f"Running RAGAS ({args.metrics}) with judge {judge_model} ...")
    df = run_ragas(rows, args.metrics, retriever.vectorstore.embedding_function)

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
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"] + "_ragas")
    with mlflow.start_run(run_name=f"ragas_{args.metrics}_{len(df)}q"):
        mlflow.log_params({
            "generator_backend": generator.name, "generator_model": generator.model_name,
            "judge_model": judge_model, "embedding_model": config["model"]["embedding_model"],
            "top_k": top_k, "num_questions": len(df), "metrics": args.metrics,
            "threshold": args.threshold,
        })
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
