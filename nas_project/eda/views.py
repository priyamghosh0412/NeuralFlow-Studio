from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
import json
from .services import handle_upload, eda_state, calculate_eda_stats, load_state_from_disk
import pandas as pd
import numpy as np

@login_required
def eda_view(request):
    """Render the EDA page."""
    return render(request, 'eda/eda.html')

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def upload_data(request):
    """Handle file upload."""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file provided'}, status=400)
            
        file = request.FILES['file']
        file_format = request.POST.get('format', 'csv').lower()
        
        result = handle_upload(file, file_format)
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def load_from_path(request):
    """Handle loading data from local path."""
    try:
        data = json.loads(request.body)
        file_path = data.get('path')
        
        if not file_path:
            return JsonResponse({'error': 'No path provided'}, status=400)
            
        from .services import load_data_from_path
        result = load_data_from_path(file_path)
        return JsonResponse(result)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["GET"])
def get_data_stats(request):
    """Get comprehensive EDA statistics."""
    load_state_from_disk()
    if eda_state['stats'] is None:
        return JsonResponse({'error': 'No data loaded'}, status=400)
    
    # Return both basic stats and comprehensive stats
    response_data = {
        **eda_state['stats'],
        'comprehensive': eda_state.get('comprehensive_stats', {})
    }
    return JsonResponse(response_data)

@csrf_exempt
@require_http_methods(["GET"])
def get_data_preview(request):
    """Get paginated data preview."""
    load_state_from_disk()
    if eda_state['dataframe'] is None:
        return JsonResponse({'error': 'No data loaded'}, status=404)
        
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 10))
        
        df = eda_state['dataframe']
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return JsonResponse({
            'data': df.iloc[start_idx:end_idx].replace({np.nan: None}).to_dict('records'),
            'total_rows': len(df),
            'page': page,
            'page_size': page_size,
            'total_pages': (len(df) + page_size - 1) // page_size
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
@csrf_exempt
@login_required
@require_http_methods(["POST"])
def db_connect(request):
    """Test database connection and return schemas."""
    try:
        data = json.loads(request.body)
        credentials = data.get('credentials')
        
        from .db_utils import test_connection, get_schemas
        success, error = test_connection(credentials)
        
        if not success:
            return JsonResponse({'success': False, 'error': error})
            
        schemas, error = get_schemas(credentials)
        if error:
            return JsonResponse({'success': False, 'error': error})
            
        return JsonResponse({'success': True, 'schemas': schemas})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def db_tables(request):
    """Get tables for a specific schema."""
    try:
        data = json.loads(request.body)
        credentials = data.get('credentials')
        schema = data.get('schema')
        
        from .db_utils import get_tables
        tables, error = get_tables(credentials, schema)
        
        if error:
            return JsonResponse({'success': False, 'error': error})
            
        return JsonResponse({'success': True, 'tables': tables})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def db_query(request):
    """Execute query and return preview."""
    try:
        data = json.loads(request.body)
        credentials = data.get('credentials')
        query = data.get('query')
        
        from .db_utils import execute_query
        df, error = execute_query(credentials, query, preview=True)
        
        if error:
            return JsonResponse({'success': False, 'error': error})
            
        # Convert to dict for JSON response
        records = df.replace({np.nan: None}).to_dict('records')
        columns = list(df.columns)
        
        return JsonResponse({
            'success': True, 
            'data': records, 
            'columns': columns,
            'rows': len(df)
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def db_load(request):
    """Execute query and load full data into EDA state."""
    try:
        data = json.loads(request.body)
        credentials = data.get('credentials')
        query = data.get('query')
        
        from .db_utils import execute_query
        df, error = execute_query(credentials, query, preview=False)
        
        if error:
            return JsonResponse({'success': False, 'error': error})
            
        # Load into EDA state
        eda_state['dataframe'] = df
        eda_state['stats'] = calculate_eda_stats(df)
        eda_state['comprehensive_stats'] = None # Will be calculated if needed or we can trigger it here
        
        # Trigger comprehensive stats calculation immediately for better UX
        from .services import calculate_comprehensive_stats
        eda_state['comprehensive_stats'] = calculate_comprehensive_stats(df)
        
        from .services import save_state_to_disk
        save_state_to_disk()
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
