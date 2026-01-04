from django.urls import path
from . import views

app_name = 'agent_studio'

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat_interface, name='chat_interface'),
    path('api/models/', views.get_models, name='get_models'),
    path('api/run/', views.run_flow, name='run_flow'),
]
