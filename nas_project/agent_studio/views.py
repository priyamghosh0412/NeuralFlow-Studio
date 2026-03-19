from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
from .utils import get_ollama_models
from .langchain_runner import LangChainRunner

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'agent_studio/index.html')

def chat_interface(request):
    return render(request, 'agent_studio/chat_interface.html')

def get_models(request):
    """API endpoint to get available running models"""
    logger.info("Fetching models from Ollama...")
    models = get_ollama_models()
    logger.info(f"Models found: {models}")
    return JsonResponse({'models': models})

@csrf_exempt
@require_http_methods(["POST"])
def run_flow(request):
    """API endpoint to execute a flow"""
    try:
        data = json.loads(request.body)
        runner = LangChainRunner()
        result = runner.run_flow(data)
        return JsonResponse({'result': result, 'status': 'success'})
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON', 'status': 'error'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e), 'status': 'error'}, status=500)
