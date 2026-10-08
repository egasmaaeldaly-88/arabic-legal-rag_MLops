import sys
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    print("Starting RAGAS Evaluation Pipeline...")

    # Define Hugging Face model IDs
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
    embedding_model_name = "BAAI/bge-m3"
    
    print("Loading Models on GPU...")
    llm = HuggingFacePipeline.from_model_id(
        model_id=llm_model_name,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0
        },
        model_kwargs={"device_map": "auto"}
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Load FAISS vector store (تأكدي من اسم المجلد لو مختلف عندك)
    print("Loading FAISS vector store...")
    vectorstore = FAISS.load_local("data/vector_store_bgem3", embeddings, allow_dangerous_deserialization=True)   
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load the JSON test set
    dataset_path = "data/eval_dataset.json"
    print(f"Loading test set from {dataset_path}...")
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Evaluation dataset not found at {dataset_path}")
        sys.exit(1)

    prepared_data = {
        "question": [],
        "answer": [],
        "contexts": []
    }

    print("Generating answers and retrieving contexts for questions...")
    for item in test_data:
        question = item["query"]
        
        # 1. Retrieve legal contexts from FAISS
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # 2. Generate answer
        prompt = f"Context: {' '.join(contexts)}\n\nQuestion: {question}\n\nAnswer:"
        answer = llm.invoke(prompt)

        # 3. Store data for RAGAS
        prepared_data["question"].append(question)
        prepared_data["contexts"].append(contexts)
        prepared_data["answer"].append(answer)

    # Convert the dictionary to a Hugging Face Dataset object
    dataset = Dataset.from_dict(prepared_data)

    # Execute RAGAS faithfulness evaluation
    print("Evaluating faithfulness metric...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings
    )
    
    df_result = result.to_pandas()
    mean_faithfulness = df_result["faithfulness"].mean()
    
    print(f"Mean Faithfulness Score: {mean_faithfulness:.2f}")

    # Enforce the 0.75 threshold requirement
    if mean_faithfulness >= 0.75:
        print("✅ Evaluation passed: Faithfulness score meets the threshold.")
        sys.exit(0)
    else:
        print("❌ Evaluation failed: Faithfulness score is below 0.75.")
        sys.exit(1)

if __name__ == "__main__":
    main()