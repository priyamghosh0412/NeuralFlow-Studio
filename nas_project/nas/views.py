from django.shortcuts import render, HttpResponse
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
import json
import os
import pandas as pd
from .services import (
    prepare_nas_data,
    start_nas_search,
    stop_nas_search,
    search_state
)
from utils.common import get_upload_folder

@login_required
def nas_view(request):
    """Render the main NAS page."""
    return render(request, 'nas/nas.html')

from transformation.services import transformation_state, ensure_state_loaded
from eda.services import eda_state

@login_required
@require_http_methods(["GET"])
def get_columns(request):
    """Get columns from the current dataframe (transformed or original)."""
    ensure_state_loaded()
    df = transformation_state['transformed_df']
    if df is None:
        df = eda_state['dataframe']
        
    if df is None:
        return JsonResponse({'error': 'No data loaded'}, status=404)
        
    return JsonResponse({'columns': list(df.columns)})

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def prepare_data(request):
    """Prepare data for NAS."""
    try:
        data = json.loads(request.body)
        target_col = data.get('target_column')
        feature_cols = data.get('feature_columns', [])
        split_ratio = data.get('split_ratio', 0.2)
        split_method = data.get('split_method', 'random')
        time_col = data.get('time_column')
        
        success, result = prepare_nas_data(target_col, feature_cols, split_ratio, split_method, time_col)
        if not success:
            return JsonResponse({'error': result}, status=400)
            
        return JsonResponse({'success': True, 'shapes': result})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def start_search(request):
    """Start NAS search."""
    try:
        data = json.loads(request.body)
        success, message = start_nas_search(data)
        if not success:
            return JsonResponse({'error': message}, status=400)
        return JsonResponse({'message': message})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def stop_search(request):
    """Stop NAS search."""
    stop_nas_search()
    return JsonResponse({'message': 'Stop signal sent'})

@login_required
@require_http_methods(["GET"])
def get_status(request):
    """Get search status."""
    return JsonResponse(search_state)

@login_required
@require_http_methods(["GET"])
def download_report(request):
    """Download search report."""
    report_path = os.path.join(get_upload_folder(), 'report.csv')
    
    if search_state['history']:
        df = pd.DataFrame(search_state['history'])
        df.to_csv(report_path, index=False)
    elif not os.path.exists(report_path):
        return JsonResponse({'error': 'No report found'}, status=404)
    
    return FileResponse(open(report_path, 'rb'), as_attachment=True, filename='nas_report.csv')

@login_required
@require_http_methods(["GET"])
def download_model(request):
    """Download best model."""
    model_path = search_state['best_model_path']
    if not model_path:
        model_path = os.path.join(get_upload_folder(), 'best_model.keras')
        
    if not os.path.exists(model_path):
        return JsonResponse({'error': 'No model found'}, status=404)
        
    return FileResponse(open(model_path, 'rb'), as_attachment=True, filename='best_model.keras')
