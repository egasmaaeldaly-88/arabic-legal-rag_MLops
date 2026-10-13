import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
c = MlflowClient()
exp = c.get_experiment_by_name("arabic-legal-rag-retrieval")
for r in c.search_runs([exp.experiment_id], max_results=1000):
    mode = "arabic_only" if r.data.params.get("overlap", "0") != "0" else "arabic_english"
    c.set_tag(r.info.run_id, "content_mode", mode)
print("tagged")