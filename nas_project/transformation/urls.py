from django.urls import path
from . import views

urlpatterns = [
    path('', views.transformation_view, name='transformation_view'),
    path('api/get_transformation_options', views.get_transformation_options, name='get_transformation_options'),
    path('api/apply_transformations', views.apply_transformation, name='apply_transformation'),
    path('api/apply_selected_batch', views.apply_selected_batch, name='apply_selected_batch'),
    path('api/get_current_preview', views.get_current_preview, name='get_current_preview'),
    path('api/chat_transform', views.chat_transform, name='chat_transform'),
    path('api/download_report', views.download_report, name='download_report'),
]
