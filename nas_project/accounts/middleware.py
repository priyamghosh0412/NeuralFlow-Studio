from utils.session_config import get_session_timeout
from django.conf import settings
import time

class ConfigurableSessionTimeoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            # Get the configured timeout in seconds
            timeout = get_session_timeout()
            
            # Set the session expiry
            # This sets the expiry timestamp in the session
            request.session.set_expiry(timeout)
            
        response = self.get_response(request)
        return response
