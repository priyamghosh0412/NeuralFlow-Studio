from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('reset-password/<int:user_id>/', views.reset_password_view, name='reset_password'),
    path('register/', views.register_view, name='register'),
    path('homepage/', views.homepage_view, name='homepage'),
    path('about/', views.about_view, name='about'),
    path('coming-soon/', views.under_construction_view, name='under_construction'),
]
