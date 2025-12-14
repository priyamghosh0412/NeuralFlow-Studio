import yaml
import os
from django.conf import settings

def get_session_timeout():
    """
    Reads the session timeout from session_config.yaml.
    Returns the timeout in seconds.
    Default is 2 hours (7200 seconds).
    """
    config_path = os.path.join(settings.BASE_DIR, 'session_config.yaml')
    default_hours = 2
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'duration_hours' in config:
                    hours = config['duration_hours']
                    # Enforce range 1-24
                    hours = max(1, min(24, int(hours)))
                    return hours * 3600
    except Exception as e:
        print(f"Error reading session config: {e}")
        
    return default_hours * 3600
