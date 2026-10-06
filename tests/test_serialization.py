from arabic_legal_rag.config import load_config

def test_config_and_vector_store_serialization_paths():
    config = load_config()
    index_dir = config["vector_db"]["index_dir"]
    assert isinstance(index_dir, str)
    assert len(index_dir) > 0

