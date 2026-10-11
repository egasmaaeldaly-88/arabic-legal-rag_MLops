import os
import sys
import json

import mlflow
import torch
from datasets import Dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- Configuration ----------------
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Use a stronger judge if your GPU allows, e.g. set JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GEN_MODEL)
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_STORE_PATH = "data/vector_store_bgem3"
DATASET_PATH = "data/eval_dataset.json"

NUM_QUESTIONS = 50
TOP_K = 3
THRESHOLD = 0.75
MIN_VALID_RATIO = 0.9  # at least 90% of rows must get a valid judgment

# Informational only: must match how the index was actually built
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

SYSTEM_PROMPT = (
    "أنت مساعد قانوني. أجب باللغة العربية وبالاعتماد فقط على السياق المقدّم. "
    "إذا لم تجد الإجابة في السياق فقل: لا أعرف."
)


def load_dataset_file(path):
    if not os.path.exists(path):
        print(f"❌ Error: Evaluation dataset not found at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) < NUM_QUESTIONS:
        print(f"❌ Error: dataset has {len(data)} items, need at least {NUM_QUESTIONS}.")
        sys.exit(1)
    return data[:NUM_QUESTIONS]


def load_llm(name):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="auto"
    )
    return tok, mdl


def build_pipe(tok, mdl, max_new_tokens):
    return pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )


def main():
    print("=== Starting RAGAS & MLflow Evaluation Pipeline ===")

    # Fail fast before loading any model
    test_data = load_dataset_file(DATASET_PATH)
    print(f"Evaluating {len(test_data)} questions...")

    passed = False

    mlflow.set_experiment("Arabic_Legal_RAG_Evaluation")
    with mlflow.start_run(run_name="ragas_faithfulness_evaluation_run"):
        mlflow.log_params({
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "llm_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "num_questions": len(test_data),
        })

        print("Loading models...")
        gen_tok, gen_model = load_llm(GEN_MODEL)
        gen_pipe = build_pipe(gen_tok, gen_model, max_new_tokens=256)

        if JUDGE_MODEL == GEN_MODEL:
            print("⚠️  Judge model == generator model. Scores may be biased; "
                  "set JUDGE_MODEL to a stronger model for a reliable CI gate.")
            judge_pipe = build_pipe(gen_tok, gen_model, max_new_tokens=1024)
        else:
            j_tok, j_model = load_llm(JUDGE_MODEL)
            judge_pipe = build_pipe(j_tok, j_model, max_new_tokens=1024)

        eval_llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=judge_pipe))
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        eval_embeddings = LangchainEmbeddingsWrapper(embeddings)

        print("Loading FAISS vector store...")
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        rows = {"user_input": [], "response": [], "retrieved_contexts": []}

        for i, item in enumerate(test_data):
            question = item["query"]
            print(f"-> Processing question {i + 1}/{len(test_data)}...")
            docs = retriever.invoke(question)
            contexts = [d.page_content for d in docs]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "السياق:\n" + "\n\n".join(contexts)
                               + f"\n\nالسؤال: {question}",
                },
            ]
            prompt = gen_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer = gen_pipe(prompt)[0]["generated_text"].strip()

            rows["user_input"].append(question)
            rows["response"].append(answer)
            rows["retrieved_contexts"].append(contexts)

        dataset = Dataset.from_dict(rows)

        print("Running RAGAS evaluation (Faithfulness)...")
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=eval_llm,
            embeddings=eval_embeddings,
        )

        df = result.to_pandas()
        n_valid = int(df["faithfulness"].notna().sum())
        valid_ratio = n_valid / len(df)
        mean_faithfulness = float(df["faithfulness"].mean())  # NaNs skipped

        print(f"\nValid judgments: {n_valid}/{len(df)}")
        print(f"Mean Faithfulness Score: {mean_faithfulness:.3f}")

        mlflow.log_metric("valid_rows", n_valid)
        mlflow.log_metric("valid_ratio", valid_ratio)
        mlflow.log_metric("ragas_faithfulness", mean_faithfulness)

        # Save per-question results for inspection
        df.to_csv("ragas_results.csv", index=False, encoding="utf-8-sig")
        mlflow.log_artifact("ragas_results.csv")

        if valid_ratio < MIN_VALID_RATIO:
            print(f"❌ Too many failed judgments ({n_valid}/{len(df)}). "
                  "Use a stronger judge model.")
            mlflow.set_tag("evaluation_status", "INVALID")
        elif mean_faithfulness >= THRESHOLD:
            print(f"✅ Passed: faithfulness >= {THRESHOLD}.")
            mlflow.set_tag("evaluation_status", "PASSED")
            passed = True
        else:
            print(f"❌ Failed: faithfulness < {THRESHOLD} (CI gate failed).")
            mlflow.set_tag("evaluation_status", "FAILED")

    # Exit only after the MLflow run has closed cleanly
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()