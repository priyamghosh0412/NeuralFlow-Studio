from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from datetime import datetime
import json
import pandas as pd
import numpy as np

def generate_transformation_report(applied_transformations, original_shape, final_shape):
    """Generate aesthetic HTML transformation report."""
    
    total_steps = len(applied_transformations)
    successful_steps = sum(1 for t in applied_transformations if t.get('status') == 'success')
    failed_steps = total_steps - successful_steps
    
    steps_html = ""
    for i, trans in enumerate(applied_transformations, 1):
        status_color = "green" if trans.get('status') == 'success' else "red"
        status_icon = "✓" if trans.get('status') == 'success' else "✗"
        
        steps_html += f"""
        <div class="border-l-4 border-{status_color}-500 bg-gray-50 p-4 mb-4 rounded-r-lg">
            <div class="flex justify-between items-start">
                <div>
                    <h3 class="font-semibold text-gray-800">Step {i}: {trans.get('title', 'Transformation')}</h3>
                    <p class="text-sm text-gray-600 mt-1">{trans.get('description', '')}</p>
                    <p class="text-xs text-gray-500 mt-2">Columns affected: {', '.join(trans.get('affected_columns', []))}</p>
                </div>
                <span class="text-2xl text-{status_color}-600">{status_icon}</span>
            </div>
            <p class="text-sm text-{status_color}-700 mt-2">{trans.get('message', '')}</p>
        </div>
        """
    
    template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Transformation Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        </style>
    </head>
    <body class="bg-gradient-to-br from-gray-50 to-gray-100 p-8">
        <div class="max-w-5xl mx-auto">
            <div class="bg-white rounded-2xl shadow-2xl overflow-hidden">
                <!-- Header -->
                <div class="bg-gradient-to-r from-indigo-600 to-purple-600 px-8 py-6 text-white">
                    <h1 class="text-4xl font-bold mb-2">Data Transformation Report</h1>
                    <p class="text-indigo-100">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div class="p-8">
                    <!-- Summary Stats -->
                    <div class="grid grid-cols-4 gap-4 mb-8">
                        <div class="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl border border-blue-200">
                            <h3 class="text-sm font-medium text-blue-600 uppercase tracking-wide">Total Steps</h3>
                            <p class="text-3xl font-bold text-blue-900 mt-2">{total_steps}</p>
                        </div>
                        <div class="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl border border-green-200">
                            <h3 class="text-sm font-medium text-green-600 uppercase tracking-wide">Successful</h3>
                            <p class="text-3xl font-bold text-green-900 mt-2">{successful_steps}</p>
                        </div>
                        <div class="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl border border-red-200">
                            <h3 class="text-sm font-medium text-red-600 uppercase tracking-wide">Failed</h3>
                            <p class="text-3xl font-bold text-red-900 mt-2">{failed_steps}</p>
                        </div>
                        <div class="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl border border-purple-200">
                            <h3 class="text-sm font-medium text-purple-600 uppercase tracking-wide">Final Rows</h3>
                            <p class="text-3xl font-bold text-purple-900 mt-2">{final_shape[0]:,}</p>
                        </div>
                    </div>
                    
                    <!-- Dataset Changes -->
                    <div class="bg-gray-50 rounded-xl p-6 mb-8 border border-gray-200">
                        <h2 class="text-xl font-bold text-gray-800 mb-4">Dataset Changes</h2>
                        <div class="grid grid-cols-2 gap-8">
                            <div>
                                <h3 class="text-sm font-medium text-gray-600 mb-2">Original Dataset</h3>
                                <p class="text-lg text-gray-800"><span class="font-bold">{original_shape[0]:,}</span> rows × <span class="font-bold">{original_shape[1]}</span> columns</p>
                            </div>
                            <div>
                                <h3 class="text-sm font-medium text-gray-600 mb-2">Transformed Dataset</h3>
                                <p class="text-lg text-gray-800"><span class="font-bold">{final_shape[0]:,}</span> rows × <span class="font-bold">{final_shape[1]}</span> columns</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Transformation Steps -->
                    <div class="mb-8">
                        <h2 class="text-2xl font-bold text-gray-800 mb-6">Transformation Pipeline</h2>
                        {steps_html}
                    </div>
                    
                    <!-- Footer -->
                    <div class="text-center text-sm text-gray-500 pt-6 border-t border-gray-200">
                        <p>Generated by NAS Framework - Data Transformation Module</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return template
