import os
from django.conf import settings

def get_upload_folder():
    """Returns the path to the upload folder."""
    return settings.MEDIA_ROOT

def ensure_upload_folder():
    """Ensures the upload folder exists."""
    os.makedirs(get_upload_folder(), exist_ok=True)
