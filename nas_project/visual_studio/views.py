from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging
import requests
import numpy as np
from eda.services import eda_state

logger = logging.getLogger(__name__)

def index(request):
    return render(request, 'visual_studio/index.html')

def get_columns(request):
    """Return columns and types from the transformed dataset."""
    if eda_state['dataframe'] is None:
        # Fallback for dev/testing if no data loaded
        # return JsonResponse({'columns': [{'name': 'test_col', 'type': 'numeric'}]})
        return JsonResponse({'error': 'No data loaded'}, status=400)
    
    df = eda_state['dataframe']
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        col_type = 'numeric' if 'int' in dtype or 'float' in dtype else 'categorical'
        if 'datetime' in dtype:
            col_type = 'datetime'
        columns.append({'name': col, 'type': col_type})
        
    return JsonResponse({'columns': columns})

def get_viz_types(request):
    """Return available visualization types."""
    viz_types = [
        {'id': 'bar', 'name': 'Bar Chart', 'icon': 'bar-chart', 'zones': ['x', 'y', 'color']},
        {'id': 'line', 'name': 'Line Chart', 'icon': 'activity', 'zones': ['x', 'y', 'color']},
        {'id': 'scatter', 'name': 'Scatter Plot', 'icon': 'circle', 'zones': ['x', 'y', 'color', 'size']},
        {'id': 'histogram', 'name': 'Histogram', 'icon': 'bar-chart-2', 'zones': ['x', 'color']},
        {'id': 'box', 'name': 'Box Plot', 'icon': 'box', 'zones': ['x', 'y', 'color']},
        {'id': 'pie', 'name': 'Pie Chart', 'icon': 'pie-chart', 'zones': ['x', 'color']},
    ]
    return JsonResponse({'viz_types': viz_types})

@csrf_exempt
@require_http_methods(["POST"])
def generate_summary(request):
    """Generate summary using local LLM."""
    try:
        data = json.loads(request.body)
        viz_config = data.get('config')
        
        if not viz_config:
            return JsonResponse({'error': 'No config provided'}, status=400)

        # Basic prompt construction
        prompt = f"You are a helpful data analyst. Analyze this plot configuration: Type: {viz_config.get('type')}. "
        prompt += f"X Axis: {viz_config.get('x')}. "
        if viz_config.get('y'):
            prompt += f"Y Axis: {viz_config.get('y')}. "
        if viz_config.get('color'):
            prompt += f"Color Group: {viz_config.get('color')}. "
            
        prompt += "Provide a brief, 2-sentence summary of what this visualization helps to understand."

        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False
        }
        
        # Connect to local ollama
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.ok:
                resp_data = response.json()
                summary = resp_data.get('response', 'No summary model output.')
                return JsonResponse({'summary': summary})
            else:
                 return JsonResponse({'summary': 'AI service unavailable.'})
        except requests.exceptions.ConnectionError:
            return JsonResponse({'summary': 'Local LLM service (Ollama) is not running.'})

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return JsonResponse({'error': str(e)}, status=500)
    
@csrf_exempt
@require_http_methods(["POST"])
def get_plot_data(request):
    """
    Fetch data for the plot. 
    Returns the actual data points for the selected columns to the frontend 
    so Plotly.js can render it.
    """
    try:
        data = json.loads(request.body)
        columns = data.get('columns', [])
        
        if eda_state['dataframe'] is None:
             return JsonResponse({'error': 'No data loaded'}, status=400)
             
        df = eda_state['dataframe']
        
        # Filter only requested columns
        valid_cols = [c for c in columns if c in df.columns]
        
        # Limit rows for performance if needed, but for 'transform' scale ~10k rows is usually fine.
        # Let's cap at 5000 for safety in this prototype
        plot_df = df[valid_cols].head(5000).replace({np.nan: None})
        
        return JsonResponse({
            'data': plot_df.to_dict('records')
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
