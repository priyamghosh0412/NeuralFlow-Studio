from django.urls import path
from . import views

urlpatterns = [
    path('', views.eda_view, name='eda_view'),
    path('api/upload_data', views.upload_data, name='upload_data'),
    path('api/load_from_path', views.load_from_path, name='load_from_path'),
    path('api/data_stats', views.get_data_stats, name='get_data_stats'),
    path('api/data_preview', views.get_data_preview, name='data_preview'),
    path('api/db/connect', views.db_connect, name='db_connect'),
    path('api/db/tables', views.db_tables, name='db_tables'),
    path('api/db/query', views.db_query, name='db_query'),
    path('api/db/load', views.db_load, name='db_load'),
    path('api/insights', views.generate_insights_view, name='generate_insights'),
    path('analysis_design/', views.analysis_design_selection_view, name='analysis_design_selection'),
    path('analysis_design/<int:option_id>/', views.analysis_design_option_view, name='analysis_design_option'),
]
