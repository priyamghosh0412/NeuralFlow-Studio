from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/columns', views.get_columns, name='get_columns'),
    path('api/viz_types', views.get_viz_types, name='get_viz_types'),
    path('api/generate_summary', views.generate_summary, name='generate_summary'),
    path('api/get_plot_data', views.get_plot_data, name='get_plot_data'),
]
