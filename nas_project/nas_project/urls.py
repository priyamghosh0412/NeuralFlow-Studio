from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('accounts.urls')),
    path('eda/', include('eda.urls')),
    path('transformation/', include('transformation.urls')),
    path('nas/', include('nas.urls')),
    path('agent-studio/', include('agent_studio.urls')),
    path('visual-studio/', include('visual_studio.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
