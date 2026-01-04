import pandas as pd
import numpy as np
from eda.services import eda_state, load_state_from_disk
from .transformation_helpers import apply_transformation, parse_chat_transformation, generate_transformation_code, generate_error_resolution
from .transformation_catalog import TRANSFORMATION_CATALOG

# Global state for Transformation
transformation_state = {
    'original_df': None,
    'transformed_df': None,
    'suggestions': [],
    'applied_transformations': [],
    'chat_history': []
}

def ensure_state_loaded():
    """Ensure EDA state is loaded and Transformation state is initialized."""
    load_state_from_disk()
    if eda_state['dataframe'] is not None and transformation_state['original_df'] is None:
        transformation_state['original_df'] = eda_state['dataframe'].copy()
        transformation_state['transformed_df'] = eda_state['dataframe'].copy()

def get_transformation_options():
    """Get transformation options based on EDA stats."""
    ensure_state_loaded()
    
    if eda_state['stats'] is None:
        return None, "No data loaded"
        
    df = transformation_state['transformed_df']
    stats = eda_state['stats']
    
    available_options = {}
    
    if stats['missing_overall'] > 0:
        available_options['missing_values'] = TRANSFORMATION_CATALOG['missing_values']
        
    if stats['duplicate_rows'] > 0:
        available_options['duplicates'] = TRANSFORMATION_CATALOG['duplicates']
        
    numerical_cols = [col for col, s in stats['columns'].items() if s['type'] == 'numerical']
    if numerical_cols:
        available_options['outliers'] = TRANSFORMATION_CATALOG['outliers']
        available_options['scaling'] = TRANSFORMATION_CATALOG['scaling']
        available_options['distribution'] = TRANSFORMATION_CATALOG['distribution']
        
    categorical_cols = [col for col, s in stats['columns'].items() if s['type'] == 'categorical']
    if categorical_cols:
        available_options['encoding'] = TRANSFORMATION_CATALOG['encoding']
        
    # Generate Column Summary with AI Suggestions
    column_summary = []
    for col in df.columns:
        col_stat = stats['columns'].get(col, {})
        
        # Fallback if stat is missing (e.g. new column)
        if not col_stat and col in df.columns:
            missing_val = df[col].isnull().sum()
            total_val = len(df)
            missing_pct = (missing_val / total_val) * 100 if total_val > 0 else 0
            unique_count = df[col].nunique()
            dtype = 'numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        else:
            missing_pct = col_stat.get('missing_pct', 0)
            unique_count = col_stat.get('unique_count', 0)
            dtype = col_stat.get('type', 'unknown')

        # AI Suggestion Logic
        suggestion = ""
        if missing_pct > 0:
            if missing_pct > 50:
                 suggestion = "Drop Column (High Missing)"
            elif dtype == 'numerical':
                suggestion = "Impute with Median/Mean"
            else:
                suggestion = "Impute with Mode or Unknown"
        elif unique_count == 1:
            suggestion = "Drop Column (Constant)"
        elif unique_count == len(df) and dtype != 'numerical':
             suggestion = "Drop Column (ID-like)"
        
        column_summary.append({
            'name': col,
            'missing_pct': round(missing_pct, 1),
            'unique_count': unique_count,
            'suggestion': suggestion
        })

    return {
        'options': available_options,
        'column_summary': column_summary,
        'columns': {
            'all': list(df.columns),
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
    }, None

def apply_transformations_logic(transformations):
    """Apply a list of transformations."""
    ensure_state_loaded()
    
    if transformation_state['transformed_df'] is None:
        return None, "No data available"
        
    if not transformations:
        return None, "No transformations provided"
        
    df = transformation_state['transformed_df'].copy()
    applied = []
    
    for trans in transformations:
        trans_id = trans.get('id')
        trans_type = trans.get('transformation_type')
        
        if trans_type == 'chat_transform':
            # Handle chat transform (simplified for brevity, full logic in app.py)
            chat_id = trans.get('chat_id')
            generated_code = trans.get('edited_code')
            
            if not generated_code and chat_id is not None and chat_id < len(transformation_state['chat_history']):
                chat_entry = transformation_state['chat_history'][chat_id]
                generated_code = chat_entry.get('generated_code')
            
            if generated_code:
                try:
                    namespace = {'df': df, 'pd': pd, 'np': np}
                    exec(generated_code, namespace)
                    df = namespace['df']
                    applied.append({'id': trans_id, 'title': 'Custom Transformation', 'timestamp': pd.Timestamp.now().isoformat()})
                except Exception as e:
                    print(f"Error in chat transform: {e}")
                    # Generate AI resolution
                    df_info = {
                        'columns': list(df.columns),
                        'rows': len(df)
                    }
                    resolution = generate_error_resolution(generated_code, str(e), df_info)
                    
                    return None, {
                        'message': str(e),
                        'resolution': resolution,
                        'failed_code': generated_code
                    }
        else:
            # Handle standard transform
            try:
                params = trans.get('params', {})
                affected_columns = trans.get('affected_columns', [])
                
                # Dynamic Column Resolution
                if affected_columns == ['__ALL_COLUMNS__']:
                    affected_columns = list(df.columns)
                elif affected_columns == ['__ALL_EXCEPT_TARGET__']:
                    target_col = params.get('target_column')
                    if target_col and target_col in df.columns:
                        affected_columns = [c for c in df.columns if c != target_col]
                    else:
                        affected_columns = list(df.columns)
                        
                params['affected_columns'] = affected_columns
                
                df, success, message = apply_transformation(df, trans_type, params)
                
                if success:
                    applied.append({
                        'id': trans_id,
                        'title': message,
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
                else:
                    raise Exception(message)
                    
            except Exception as e:
                return None, f"Error applying {trans_type}: {str(e)}"
                
    transformation_state['transformed_df'] = df
    transformation_state['applied_transformations'].extend(applied)
    
    return {
        'success': True,
        'applied': applied,
        'preview': df.head(10).replace({np.nan: None}).to_dict(orient='records'),
        'columns': list(df.columns)
    }, None

def handle_chat_transform(user_message):
    """Handle chat transformation request."""
    ensure_state_loaded()
    
    if transformation_state['transformed_df'] is None:
        return None, "No data available"
        
    df = transformation_state['transformed_df']
    df_info = {
        'columns': df.columns.tolist(),
        'rows': len(df),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    try:
        # Prepare context for code generation
        # Ensure we don't truncate columns in the string representation if possible, 
        # but for very wide datasets we might need to be careful. 
        # User requested "all columns", so we try to show them.
        df_head = df.head(5).to_string()
        df_dtypes = df.dtypes.to_string()
        
        result = parse_chat_transformation(user_message, df_info, df_head)
        
        generated_code = generate_transformation_code(user_message, df.columns.tolist(), df_head, df_dtypes)
        
        if generated_code:
            result['generated_code'] = generated_code
            result['code_preview'] = generated_code
        else:
            result['generated_code'] = None
            result['code_preview'] = "# Could not generate code"
            
        chat_entry = {
            'user_message': user_message,
            'ai_response': result,
            'generated_code': generated_code,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        transformation_state['chat_history'].append(chat_entry)
        
        return {
            'success': True,
            'response': result,
            'chat_id': len(transformation_state['chat_history']) - 1
        }, None
        
    except Exception as e:
        return None, str(e)
