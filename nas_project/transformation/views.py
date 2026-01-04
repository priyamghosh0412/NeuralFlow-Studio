from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
import json
import pandas as pd
import numpy as np
from .services import (
    get_transformation_options as fetch_transformation_options,
    apply_transformations_logic,
    handle_chat_transform,
    transformation_state,
    ensure_state_loaded
)
from .report_generator import generate_transformation_report

@login_required
def transformation_view(request):
    """Render the main transformation page."""
    context = {
        'studio_mode': request.session.get('studio_mode', 'autodl')
    }
    return render(request, 'transformation/transformation.html', context)

@login_required
@require_http_methods(["GET"])
def get_transformation_options(request):
    """Get transformation options."""
    data, error = fetch_transformation_options()
    if error:
        return JsonResponse({'error': error}, status=400)
    return JsonResponse(data)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def apply_transformation(request):
    """Apply transformations."""
    try:
        data = json.loads(request.body)
        transformations = data.get('transformations', [])
        
        result, error = apply_transformations_logic(transformations)
        if error:
            return JsonResponse({'error': error}, status=500)
            
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def apply_selected_batch(request):
    """Apply multiple transformations in sequence with progress tracking."""
    try:
        ensure_state_loaded()
        data = json.loads(request.body)
        transformations = data.get('transformations', [])
        
        if not transformations:
            return JsonResponse({'error': 'No transformations provided'}, status=400)
        
        # Store original shape for report
        original_df = transformation_state['original_df']
        if original_df is None:
            return JsonResponse({'error': 'No data available'}, status=400)
        
        original_shape = original_df.shape
        df = transformation_state['transformed_df'].copy()
        results = []
        
        # Process each transformation sequentially
        for i, trans in enumerate(transformations):
            try:
                # --- DYNAMIC COLUMN RESOLUTION ---
                # Resolve actual affected columns based on CURRENT dataframe state
                column_mode = trans.get('params', {}).get('column_mode', 'all')
                selected_columns = trans.get('params', {}).get('selected_columns', [])
                
                current_columns = list(df.columns)
                affected_columns = []
                
                if column_mode == 'all' or trans.get('affected_columns') == ['__ALL_COLUMNS__']:
                    affected_columns = current_columns
                elif column_mode == 'include':
                    # Only include columns that exist in current dataframe
                    affected_columns = [c for c in selected_columns if c in current_columns]
                elif column_mode == 'exclude':
                    # Exclude specified columns, include all others (including newly created ones)
                    affected_columns = [c for c in current_columns if c not in selected_columns]
                else:
                    # Fallback to provided affected_columns
                    affected_columns = trans.get('affected_columns', current_columns)
                
                # Update transformation with resolved columns
                trans['affected_columns'] = affected_columns
                trans_list = [trans]
                
                # Apply the transformation
                result, error = apply_transformations_logic(trans_list)
                
                if error:
                    error_msg = error
                    resolution = None
                    failed_code = None
                    
                    if isinstance(error, dict):
                        error_msg = error.get('message', 'Unknown error')
                        resolution = error.get('resolution')
                        failed_code = error.get('failed_code')
                    
                    results.append({
                        'step': i,
                        'name': trans.get('name', f'Step {i+1}'),
                        'title': trans.get('title', 'Transformation'),
                        'status': 'failed',
                        'message': error_msg,
                        'resolution': resolution,
                        'failed_code': failed_code,
                        'description': trans.get('description', ''),
                        'affected_columns': affected_columns
                    })
                    break  # Stop on first error
                else:
                    # Update df to current state for next iteration
                    df = transformation_state['transformed_df'].copy()
                    
                    results.append({
                        'step': i,
                        'name': trans.get('name', f'Step {i+1}'),
                        'title': trans.get('title', 'Transformation'),
                        'status': 'success',
                        'message': f"Applied to {len(affected_columns)} column(s): {', '.join(affected_columns[:5])}" + ('...' if len(affected_columns) > 5 else ''),
                        'description': trans.get('description', ''),
                        'affected_columns': affected_columns
                    })
            except Exception as e:
                results.append({
                    'step': i,
                    'name': trans.get('name', f'Step {i+1}'),
                    'title': trans.get('title', 'Transformation'),
                    'status': 'failed',
                    'message': str(e),
                    'description': trans.get('description', ''),
                    'affected_columns': []
                })
                break
        
        # Get final preview
        final_df = transformation_state['transformed_df']
        final_shape = final_df.shape if final_df is not None else original_shape
        
        # Store report data in session for download
        transformation_state['last_report_data'] = {
            'applied_transformations': results,
            'original_shape': original_shape,
            'final_shape': final_shape
        }
        
        return JsonResponse({
            'success': True,
            'results': results,
            'final_preview': {
                'preview': final_df.head(10).replace({np.nan: None}).to_dict(orient='records') if final_df is not None else [],
                'columns': list(final_df.columns) if final_df is not None else []
            },
            'original_preview': {
                'preview': original_df.head(10).replace({np.nan: None}).to_dict(orient='records'),
                'columns': list(original_df.columns)
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
@require_http_methods(["GET"])
def get_current_preview(request):
    """Get current transformed dataframe preview."""
    ensure_state_loaded()
    if transformation_state['transformed_df'] is None:
        return JsonResponse({'error': 'No data available'}, status=404)
        
    df = transformation_state['transformed_df']
    original_df = transformation_state['original_df']
    
    response_data = {
        'success': True,
        'preview': df.head(10).replace({float('nan'): None}).to_dict(orient='records'),
        'columns': list(df.columns),
    }
    
    if original_df is not None:
        response_data['original_preview'] = original_df.head(10).replace({float('nan'): None}).to_dict(orient='records')
        response_data['original_columns'] = list(original_df.columns)
        
    return JsonResponse(response_data)

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def chat_transform(request):
    """Handle chat transformation."""
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        
        result, error = handle_chat_transform(message)
        if error:
            return JsonResponse({'error': error}, status=500)
            
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def download_report(request):
    """Generate and download transformation report as HTML."""
    try:
        report_data = transformation_state.get('last_report_data')
        
        if not report_data:
            return HttpResponse("No transformation report available", status=404)
        
        report_html = generate_transformation_report(
            applied_transformations=report_data['applied_transformations'],
            original_shape=report_data['original_shape'],
            final_shape=report_data['final_shape']
        )
        
        from datetime import datetime
        response = HttpResponse(report_html, content_type='text/html')
        response['Content-Disposition'] = f'attachment; filename="transformation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html"'
        return response
    except Exception as e:
        return HttpResponse(f"Error generating report: {str(e)}", status=500)
