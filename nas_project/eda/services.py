import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import skew, kurtosis
from utils.common import get_upload_folder

# Global state for EDA
eda_state = {
    'dataframe': None,
    'stats': None,
    'file_info': None,
    'comprehensive_stats': None
}

def load_state_from_disk():
    """
    Deprecated: State persistence disabled per user request.
    Data is now kept in-memory only for the duration of the server session.
    """
    pass

def calculate_eda_stats(df):
    """Calculate comprehensive EDA statistics"""
    # Calculate data quality score (0-100)
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    quality_score = completeness
    
    duplicate_count = df.duplicated().sum()
    high_missing_cols = sum(1 for col in df.columns if (df[col].isnull().sum() / len(df) * 100) > 50)
    
    numerical_count = sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))
    categorical_count = len(df.columns) - numerical_count
    
    stats_dict = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'missing_overall': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        'quality_score': round(quality_score, 1),
        'duplicate_rows': int(duplicate_count),
        'high_missing_cols': int(high_missing_cols),
        'numerical_count': int(numerical_count),
        'categorical_count': int(categorical_count),
        'total_missing_cells': int(df.isnull().sum().sum()),
        'columns': {}
    }
    
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isnull().sum()),
            'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
            'unique_count': int(df[col].nunique())
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats['type'] = 'numerical'
            
            def safe_float(val):
                return None if pd.isna(val) else float(val)
            
            col_stats['stats'] = {
                'min': safe_float(df[col].min()) if not df[col].isnull().all() else None,
                'max': safe_float(df[col].max()) if not df[col].isnull().all() else None,
                'mean': safe_float(df[col].mean()) if not df[col].isnull().all() else None,
                'median': safe_float(df[col].median()) if not df[col].isnull().all() else None,
                'std': safe_float(df[col].std()) if not df[col].isnull().all() else None,
                'q25': safe_float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                'q75': safe_float(df[col].quantile(0.75)) if not df[col].isnull().all() else None
            }
            hist, bin_edges = np.histogram(df[col].dropna(), bins=20)
            col_stats['histogram'] = {
                'counts': hist.tolist(),
                'bins': [safe_float(x) for x in bin_edges.tolist()]
            }
        else:
            col_stats['type'] = 'categorical'
            value_counts = df[col].value_counts().head(3)
            col_stats['top_values'] = [
                {'value': str(v), 'count': int(c)} 
                for v, c in zip(value_counts.index.tolist(), value_counts.values.tolist())
            ]
        
        stats_dict['columns'][col] = col_stats
    
    return stats_dict

def calculate_comprehensive_stats(df):
    """Calculate comprehensive statistics for EDA dashboard (50+ metrics)"""
    
    comp_stats = {
        'overview': {},
        'columns': {},
        'missing': {},
        'correlations': {},
        'quality': {}
    }
    
    # 1. Dataset Overview
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object', 'category'])
    datetime_df = df.select_dtypes(include=['datetime'])
    
    total_cols = len(df.columns)
    comp_stats['overview'] = {
        'total_rows': len(df),
        'total_columns': total_cols,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        'numeric_count': len(numeric_df.columns),
        'categorical_count': len(categorical_df.columns),
        'datetime_count': len(datetime_df.columns),
        'numeric_pct': round((len(numeric_df.columns) / total_cols) * 100, 1) if total_cols > 0 else 0,
        'categorical_pct': round((len(categorical_df.columns) / total_cols) * 100, 1) if total_cols > 0 else 0,
        'datetime_pct': round((len(datetime_df.columns) / total_cols) * 100, 1) if total_cols > 0 else 0,
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_rows_pct': round((df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 0,
        'constant_columns': int(sum(df.nunique() == 1))
    }
    
    # 2. Per-Column Analysis
    for col in df.columns:
        col_data = {
            'name': col,
            'dtype': str(df[col].dtype),
            'memory_bytes': int(df[col].memory_usage(deep=True)),
            'unique_count': int(df[col].nunique()),
            'missing_count': int(df[col].isnull().sum()),
            'missing_pct': round((df[col].isnull().sum() / len(df)) * 100, 2)
        }
        
        # Cardinality level
        unique_count = col_data['unique_count']
        if unique_count < 10:
            col_data['cardinality'] = 'Low'
        elif unique_count < 50:
            col_data['cardinality'] = 'Medium'
        else:
            col_data['cardinality'] = 'High'
        
        # Numeric statistics
        if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                Q1 = non_null.quantile(0.25)
                Q3 = non_null.quantile(0.75)
                IQR = Q3 - Q1
                
                col_data['numeric'] = {
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()),
                    'variance': float(non_null.var()),
                    'skewness': float(skew(non_null)),
                    'kurtosis': float(kurtosis(non_null)),
                    'percentiles': {
                        'p1': float(non_null.quantile(0.01)),
                        'p5': float(non_null.quantile(0.05)),
                        'p25': float(Q1),
                        'p50': float(non_null.quantile(0.50)),
                        'p75': float(Q3),
                        'p95': float(non_null.quantile(0.95)),
                        'p99': float(non_null.quantile(0.99))
                    },
                    'iqr': float(IQR),
                    'cv': float(non_null.std() / non_null.mean()) if non_null.mean() != 0 else 0,
                    'zeros_count': int((non_null == 0).sum()),
                    'zeros_pct': round((non_null == 0).sum() / len(non_null) * 100, 2)
                }
                
                # Outliers (IQR method)
                outliers = ((non_null < (Q1 - 1.5 * IQR)) | (non_null > (Q3 + 1.5 * IQR))).sum()
                col_data['numeric']['outliers_count'] = int(outliers)
                col_data['numeric']['outliers_pct'] = round(outliers / len(non_null) * 100, 2)
        
        # Categorical statistics
        elif df[col].dtype in ['object', 'category']:
            value_counts = df[col].value_counts()
            total_non_null = value_counts.sum()
            
            # Top 5 frequencies
            top_5 = value_counts.head(5)
            top_5_dict = {}
            top_5_pct_dict = {}
            for val, count in top_5.items():
                key = str(val)[:50]  # Truncate long values
                top_5_dict[key] = int(count)
                top_5_pct_dict[key] = round((count / total_non_null) * 100, 2)
            
            col_data['categorical'] = {
                'unique_categories': len(value_counts),
                'top_5': top_5_dict,
                'top_5_pct': top_5_pct_dict,
                'rare_count': int((value_counts / total_non_null < 0.01).sum()),  # <1%
                'dominant_category': bool(value_counts.iloc[0] / total_non_null > 0.5) if len(value_counts) > 0 else False,
                'imbalance_ratio': round(value_counts.iloc[0] / value_counts.iloc[-1], 2) if len(value_counts) > 1 else 1.0
            }
        
        # Datetime statistics
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                col_data['datetime'] = {
                    'min_date': str(non_null.min()),
                    'max_date': str(non_null.max()),
                    'range_days': (non_null.max() - non_null.min()).days,
                }
                
                # Frequency detection
                if len(non_null) > 1:
                    diffs = non_null.sort_values().diff().dt.days.dropna()
                    if len(diffs) > 0:
                        mode_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else 0
                        if mode_diff == 1:
                            col_data['datetime']['frequency'] = 'Daily'
                        elif 6 <= mode_diff <= 8:
                            col_data['datetime']['frequency'] = 'Weekly'
                        elif 28 <= mode_diff <= 31:
                            col_data['datetime']['frequency'] = 'Monthly'
                        else:
                            col_data['datetime']['frequency'] = f'{int(mode_diff)} days'
                        
                        col_data['datetime']['avg_gap_days'] = round(diffs.mean(), 2)
        
        comp_stats['columns'][col] = col_data
    
    # 3. Missing Data Analysis
    missing_per_row = df.isnull().sum(axis=1) / len(df.columns)
    comp_stats['missing'] = {
        'rows_50pct_missing': int((missing_per_row > 0.5).sum()),
        'rows_80pct_missing': int((missing_per_row > 0.8).sum()),
        'cols_50pct_missing': int(sum(df.isnull().sum() / len(df) > 0.5))
    }
    
    # 4. Numeric Correlations
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8 and not pd.isna(corr_val):
                    high_corr_pairs.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': round(float(corr_val), 3)
                    })
        
        # Limit to top 10
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]
        
        comp_stats['correlations'] = {
            'matrix': {col: {c: round(float(v), 3) if not pd.isna(v) else None 
                            for c, v in corr_matrix[col].items()} 
                      for col in corr_matrix.columns},
            'high_corr_pairs': high_corr_pairs
        }
    else:
        comp_stats['correlations'] = {'matrix': {}, 'high_corr_pairs': []}
    
    # 5. Quality Checks
    zero_heavy_cols = []
    for col in numeric_df.columns:
        if (numeric_df[col] == 0).sum() / len(numeric_df) > 0.5:
            zero_heavy_cols.append(col)
    
    comp_stats['quality'] = {
        'duplicate_rows_pct': round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
        'zero_heavy_cols': zero_heavy_cols,
        'zero_heavy_count': len(zero_heavy_cols)
    }
    
    return comp_stats

def handle_upload(file, file_format):
    """Handle file upload and update state."""
    # Use system temp directory for transient upload handling
    import tempfile
    
    # Save to a temporary file
    suffix = f'.{file_format}'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in file.chunks():
            tmp.write(chunk)
        temp_path = tmp.name
            
    # Load dataframe
    try:
        if file_format == 'csv':
            df = pd.read_csv(temp_path)
        elif file_format in ['parquet', 'pq']:
            df = pd.read_parquet(temp_path)
        elif file_format in ['excel', 'xlsx', 'xls']:
            df = pd.read_excel(temp_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        # Update state
        eda_state['dataframe'] = df
        eda_state['stats'] = calculate_eda_stats(df)
        eda_state['comprehensive_stats'] = calculate_comprehensive_stats(df)
        eda_state['file_info'] = {
            'name': file.name,
            'size_mb': file.size / (1024 * 1024),
            'format': file_format,
            'source': 'upload'
        }
        
        return {
            'success': True,
            'file_info': eda_state['file_info'],
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'preview': df.head(10).replace({np.nan: None}).to_dict('records'),
            'columns': list(df.columns)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
        
    finally:
        # Clean up temp file immediately
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_data_from_path(file_path):
    """Load data directly from a local file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Determine format from extension
    _, ext = os.path.splitext(file_path)
    file_format = ext.lower().lstrip('.')
    
    # Load dataframe
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format in ['parquet', 'pq']:
        df = pd.read_parquet(file_path)
    elif file_format in ['excel', 'xlsx', 'xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    # Update state
    eda_state['dataframe'] = df
    eda_state['stats'] = calculate_eda_stats(df)
    eda_state['comprehensive_stats'] = calculate_comprehensive_stats(df)
    eda_state['file_info'] = {
        'name': os.path.basename(file_path),
        'size_mb': os.path.getsize(file_path) / (1024 * 1024),
        'format': file_format,
        'source': 'local_path',
        'path': file_path
    }
    
    return {
        'success': True,
        'file_info': eda_state['file_info'],
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'preview': df.head(10).replace({np.nan: None}).to_dict('records'),
        'columns': list(df.columns)
    }
