import os
import json
import re
import yaml

def normalize_arabic(text: str) -> str:
    """Standardize Arabic characters and remove diacritics for vector search consistency."""
    if not text:
        return ""
    # Remove Tashkeel (diacritics)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # Unify Alef variations
    text = re.sub(r"[إأآ]", "ا", text)
    # Unify Ya / Alef Maqsoora
    text = re.sub(r"ى", "ي", text)
    # Unify Teh Marbouta
    text = re.sub(r"ة", "ه", text)
    return text.strip()

def load_config(config_path: str = "configs/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_clean_corpus(json_path: str) -> list:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Corpus file not found at '{json_path}'")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)