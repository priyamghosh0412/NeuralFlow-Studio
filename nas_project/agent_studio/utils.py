import requests
import logging

logger = logging.getLogger(__name__)

def get_ollama_models():
    """
    Fetches the list of available models from the local Ollama instance.
    Uses the logic recommended for robust size calculation and information retrieval.
    """
    url = "http://127.0.0.1:11434/api/tags"
    logger.info(f"get_ollama_models: Fetching from {url}")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for m in data.get("models", []):
            size_gb = round(m.get('size', 0) / (1024**3), 2)
            models.append({
                'name': m['name'],
                'size': f"{size_gb} GB"
            })
        logger.info(f"get_ollama_models: Found {len(models)} models.")
        return models
    except Exception as e:
        logger.error(f"Ollama detection error: {e}")
        return []
