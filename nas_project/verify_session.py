import os
import django
from django.conf import settings
from django.test import RequestFactory
from django.contrib.sessions.middleware import SessionMiddleware
from accounts.middleware import ConfigurableSessionTimeoutMiddleware
from utils.session_config import get_session_timeout
import time

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'nas_project.settings')
django.setup()

def test_session_timeout_logic():
    print("Testing Session Timeout Logic...")
    
    # 1. Test Config Reading
    timeout = get_session_timeout()
    print(f"Current configured timeout: {timeout} seconds")
    
    # 2. Test Middleware
    factory = RequestFactory()
    request = factory.get('/')
    
    # Add session to request
    middleware = SessionMiddleware(lambda r: None)
    middleware.process_request(request)
    request.session.save()
    
    # Simulate logged in user
    class MockUser:
        is_authenticated = True
    request.user = MockUser()
    
    # Run our middleware
    timeout_middleware = ConfigurableSessionTimeoutMiddleware(lambda r: None)
    timeout_middleware(request)
    
    # Check expiry
    expiry = request.session.get_expiry_age()
    print(f"Session expiry age after middleware: {expiry}")
    
    if abs(expiry - timeout) < 10:
        print("SUCCESS: Session expiry matches configuration.")
    else:
        print(f"FAILURE: Session expiry {expiry} does not match config {timeout}.")

if __name__ == "__main__":
    test_session_timeout_logic()
