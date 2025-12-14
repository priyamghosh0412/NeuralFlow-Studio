import os
import threading
import pandas as pd
import numpy as np
from django.conf import settings
from nas_rl.core import NASFramework
from eda.services import eda_state, load_state_from_disk
from transformation.services import transformation_state, ensure_state_loaded
from sklearn.model_selection import train_test_split
from utils.common import get_upload_folder

# Global State for NAS
search_state = {
    'running': False,
    'history': [],
    'current_status': {},
    'best_model_path': None,
    'input_shape': None,
    'output_shape': None,
    'best_reward': None
}

stop_event = threading.Event()
nas_instance = None

def prepare_nas_data(target_col, feature_cols, split_ratio=0.2, split_method='random', time_col=None):
    """Prepare data for NAS search."""
    # Ensure state is loaded
    load_state_from_disk()
    ensure_state_loaded()
    
    # Use transformed DF if available, else original
    df = transformation_state['transformed_df']
    if df is None:
        df = eda_state['dataframe']
        
    if df is None:
        return False, "No data available"
        
    try:
        # Filter features
        if feature_cols:
            # Ensure target is not in features
            feature_cols = [c for c in feature_cols if c != target_col]
            # Ensure time column is preserved if needed for splitting
            cols_to_keep = feature_cols + [target_col]
            if split_method == 'time' and time_col and time_col not in cols_to_keep:
                cols_to_keep.append(time_col)
            
            df_subset = df[cols_to_keep].copy()
        else:
            df_subset = df.copy()
            
        # Handle Time-based Split
        if split_method == 'time':
            if not time_col or time_col not in df.columns:
                return False, "Time column required for time-based split"
            
            # Sort by time
            df_subset = df_subset.sort_values(by=time_col)
            
            # Calculate split index
            split_idx = int(len(df_subset) * (1 - split_ratio))
            
            train_df = df_subset.iloc[:split_idx]
            val_df = df_subset.iloc[split_idx:]
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
            
        else:
            # Random Split
            X = df_subset[feature_cols]
            y = df_subset[target_col]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=42)
        
        # Handle categorical data (simple encoding if not already done)
        # For now, assume user handled it or we select only numeric for X
        X_train = X_train.select_dtypes(include=[np.number])
        X_val = X_val.select_dtypes(include=[np.number])
        
        if not pd.api.types.is_numeric_dtype(y_train):
            # Simple label encoding for target if categorical
            y_train = y_train.astype('category').cat.codes
            y_val = y_val.astype('category').cat.codes
            
        upload_folder = get_upload_folder()
        X_train.to_csv(os.path.join(upload_folder, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(upload_folder, 'y_train.csv'), index=False)
        X_val.to_csv(os.path.join(upload_folder, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(upload_folder, 'y_val.csv'), index=False)
        
        return True, {
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_test_shape': X_val.shape,
            'y_test_shape': y_val.shape
        }
    except Exception as e:
        return False, str(e)

def run_search(X_train_path, y_train_path, X_val_path, y_val_path, target_metric, patience, problem_type):
    """Run the NAS search in a background thread."""
    global nas_instance, search_state
    
    try:
        # Force CPU to avoid MacOS MPS hangs
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass

        X_train = pd.read_csv(X_train_path).values
        y_train = pd.read_csv(y_train_path).values
        X_val = pd.read_csv(X_val_path).values
        y_val = pd.read_csv(y_val_path).values
        
        # Determine shapes
        input_shape = (X_train.shape[1],)
        if problem_type == 'classification':
            num_classes = len(np.unique(y_train))
            output_shape = num_classes if num_classes > 2 else 1
        else:
            output_shape = 1
            
        search_state['input_shape'] = input_shape[0]
        search_state['output_shape'] = output_shape
        
        nas_instance = NASFramework(
            input_shape=input_shape,
            output_shape=(output_shape,),
            problem_type=problem_type
        )
        
        def callback(status):
            search_state['current_status'] = status
            # Only append to history if it's a completed episode (has reward)
            if 'reward' in status:
                search_state['history'].append(status)
            search_state['best_reward'] = status.get('best_reward')
            
        nas_instance.search(
            X_train, y_train, X_val, y_val,
            target_metric=float(target_metric),
            patience=int(patience),
            min_episodes=int(patience),
            strategy='rl',
            callback=callback,
            stop_signal=lambda: stop_event.is_set()
        )
        
    except Exception as e:
        print(f"Search Error: {e}")
    finally:
        if nas_instance:
            best_model = nas_instance.get_best_model()
            if best_model:
                model_path = os.path.join(get_upload_folder(), 'best_model.keras')
                best_model.save(model_path)
                search_state['best_model_path'] = model_path
        
        search_state['running'] = False

def start_nas_search(config):
    """Start the NAS search."""
    global search_state, stop_event
    
    if search_state['running']:
        return False, "Search is already running."
    
    stop_event.clear()
    stop_event.clear()
    # Reset state in place to preserve reference
    search_state['running'] = True
    search_state['history'] = []
    search_state['current_status'] = {}
    search_state['best_model_path'] = None
    search_state['input_shape'] = None
    search_state['output_shape'] = None
    search_state['best_reward'] = None
    
    upload_folder = get_upload_folder()
    paths = {
        'X_train_path': os.path.join(upload_folder, 'X_train.csv'),
        'y_train_path': os.path.join(upload_folder, 'y_train.csv'),
        'X_val_path': os.path.join(upload_folder, 'X_val.csv'),
        'y_val_path': os.path.join(upload_folder, 'y_val.csv')
    }
    
    thread = threading.Thread(target=run_search, kwargs={
        **paths,
        'target_metric': config.get('target_metric', 0.95),
        'patience': config.get('patience', 5),
        'problem_type': config.get('problem_type', 'classification')
    })
    thread.start()
    
    return True, "Search started successfully."

def stop_nas_search():
    """Stop the NAS search."""
    stop_event.set()
    return True
