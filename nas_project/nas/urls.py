from django.urls import path
from . import views

urlpatterns = [
    path('', views.nas_view, name='nas_view'),
    path('api/prepare_data', views.prepare_data, name='prepare_data'),
    path('api/start', views.start_search, name='start_search'),
    path('api/stop', views.stop_search, name='stop_search'),
    path('api/status', views.get_status, name='get_status'),
    path('api/columns', views.get_columns, name='get_columns'),
    path('download_report', views.download_report, name='download_report'),
    path('download_model', views.download_model, name='download_model'),
]
