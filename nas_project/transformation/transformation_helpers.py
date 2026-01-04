"""
Helper functions for data transformations
"""
import requests
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer
import traceback

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "codellama:7b"  # Optimized for code generation tasks


def call_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama API with a prompt"""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            return None
    except Exception as e:
        print(f"Ollama API error: {e}")
        return None


def generate_data_insights(eda_stats):
    """Generate deterministic insights based on EDA statistics"""
    insights = []
    
    # 1. Dataset Shape
    rows = eda_stats['shape']['rows']
    cols = eda_stats['shape']['columns']
    insights.append(f"The dataset consists of **{rows:,} rows** and **{cols} columns**.")
    
    # 2. Missing Values
    missing_cols = [col for col, stats in eda_stats['columns'].items() if stats['missing_pct'] > 0]
    if missing_cols:
        highest_missing = sorted(missing_cols, key=lambda x: eda_stats['columns'][x]['missing_pct'], reverse=True)[0]
        pct = eda_stats['columns'][highest_missing]['missing_pct']
        insights.append(f"**{len(missing_cols)} columns** contain missing values. **{highest_missing}** has the most ({pct:.1f}%).")
    else:
        insights.append("Data quality is excellent with **no missing values** detected.")
        
    # 3. Duplicates
    duplicates = eda_stats.get('duplicate_rows', 0)
    if duplicates > 0:
        insights.append(f"Found **{duplicates:,} duplicate rows** ({duplicates/rows*100:.1f}%) that may need cleaning.")
    else:
        insights.append("The dataset is free of **duplicate rows**.")
        
    # 4. Data Types Distribution
    types = [stats['type'] for stats in eda_stats['columns'].values()]
    numeric_count = types.count('numerical')
    categorical_count = types.count('categorical')
    insights.append(f"Feature mix: **{numeric_count} numerical** and **{categorical_count} categorical** variables.")
    
    # 5. High Cardinality or Constant Columns
    constant_cols = [col for col, stats in eda_stats['columns'].items() if stats['unique_count'] == 1]
    if constant_cols:
        insights.append(f"**{len(constant_cols)} constant columns** detected (e.g., {constant_cols[0]}) which provide no information.")
    
    return insights


def generate_transformation_suggestions(eda_stats):
    """Generate transformation suggestions based on EDA statistics"""
    
    # Prepare detailed column information
    column_details = []
    for col, stats in eda_stats['columns'].items():
        col_info = {
            'name': col,
            'type': stats['type'],
            'dtype': stats['dtype'],
            'missing_pct': stats['missing_pct'],
            'unique_count': stats['unique_count']
        }
        
        # Add type-specific stats
        if stats['type'] == 'numerical' and stats.get('stats'):
            col_info['min'] = stats['stats'].get('min')
            col_info['max'] = stats['stats'].get('max')
            col_info['mean'] = stats['stats'].get('mean')
            col_info['median'] = stats['stats'].get('median')
        elif stats['type'] == 'categorical' and stats.get('top_values'):
            col_info['top_values'] = [f"{v['value']} ({v['count']})" for v in stats['top_values'][:3]]
        
        column_details.append(col_info)
    
    prompt = f"""You are a data cleaning expert analyzing a real dataset. Based on the ACTUAL column-level statistics below, suggest SPECIFIC, ACTIONABLE data transformations.

DATASET OVERVIEW:
- Total Rows: {eda_stats['shape']['rows']:,}
- Total Columns: {eda_stats['shape']['columns']}
- Overall Missing: {eda_stats['missing_overall']:.2f}%
- Duplicate Rows: {eda_stats['duplicate_rows']:,}
- Data Quality Score: {eda_stats['quality_score']}/100

DETAILED COLUMN STATISTICS:
{json.dumps(column_details, indent=2)}

CRITICAL RULES:
1. Analyze the ACTUAL columns listed above
2. CONSOLIDATE similar transformations - DO NOT create separate suggestions for each column pair
3. For missing values: Create ONE suggestion per strategy (e.g., one for "median imputation", one for "drop rows"), listing ALL affected columns
4. Be SPECIFIC - mention exact column names in descriptions
5. Provide DIVERSE transformation types (missing values, duplicates, encoding, scaling, outliers)
6. Valid transformation types ONLY: handle_missing, remove_duplicates, encode_categorical, scale_numerical, remove_outliers
7. Maximum 5-6 suggestions total - NO redundant suggestions

REQUIRED FORMAT (JSON array):
[
  {{
    "id": "unique_id",
    "title": "Specific transformation title",
    "description": "Detailed description mentioning ALL affected columns",
    "affected_columns": ["col1", "col2", "col3", "col4"],
    "impact": "low|medium|high",
    "transformation_type": "handle_missing|remove_duplicates|encode_categorical|scale_numerical|remove_outliers"
  }}
]

Generate 5-6 DIVERSE, NON-REDUNDANT suggestions. Return ONLY the JSON array."""

    response = call_ollama(prompt)
    
    if response:
        try:
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                suggestions = json.loads(json_str)
                
                # Validate suggestions
                valid_types = ['handle_missing', 'remove_duplicates', 'encode_categorical', 
                              'scale_numerical', 'remove_outliers', 'feature_engineering']
                valid_suggestions = []
                
                for s in suggestions:
                    # Check if transformation type is valid
                    if s.get('transformation_type') not in valid_types:
                        continue
                    
                    # Check if affected columns are real
                    affected = s.get('affected_columns', [])
                    if affected and affected != ['all']:
                        # Verify at least one column exists
                        real_cols = [col for col in affected if col in eda_stats['columns']]
                        if not real_cols:
                            continue
                        s['affected_columns'] = real_cols  # Use only real columns
                    
                    valid_suggestions.append(s)
                
                if valid_suggestions:
                    return valid_suggestions
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            pass
    
    # Fallback: Generate comprehensive suggestions
    return generate_fallback_suggestions(eda_stats)


def generate_fallback_suggestions(eda_stats):
    """Generate comprehensive, data-aware transformation suggestions"""
    suggestions = []
    
    # Get columns with actual missing values
    cols_with_missing = {
        col: stats for col, stats in eda_stats['columns'].items() 
        if stats['missing_pct'] > 0
    }
    
    # Separate by type
    numerical_missing = [
        col for col, stats in cols_with_missing.items() 
        if stats['type'] == 'numerical'
    ]
    categorical_missing = [
        col for col, stats in cols_with_missing.items() 
        if stats['type'] == 'categorical'
    ]
    
    # Strategy 1: Median imputation for numerical columns with missing values
    if numerical_missing:
        suggestions.append({
            "id": "handle_missing_median",
            "title": f"Fill Missing Values with Median",
            "description": f"Impute missing values in {len(numerical_missing)} numerical column(s): {', '.join(numerical_missing[:3])}{'...' if len(numerical_missing) > 3 else ''} using median (robust to outliers)",
            "affected_columns": numerical_missing,
            "impact": "medium",
            "transformation_type": "handle_missing"
        })
    
    # Strategy 2: Mean imputation for numerical columns
    if numerical_missing:
        suggestions.append({
            "id": "handle_missing_mean",
            "title": f"Fill Missing Values with Mean",
            "description": f"Impute missing values in {len(numerical_missing)} numerical column(s): {', '.join(numerical_missing[:3])}{'...' if len(numerical_missing) > 3 else ''} using mean",
            "affected_columns": numerical_missing,
            "impact": "medium",
            "transformation_type": "handle_missing"
        })
    
    # Strategy 3: Mode imputation for categorical columns
    if categorical_missing:
        suggestions.append({
            "id": "handle_missing_mode",
            "title": f"Fill Missing Values with Mode",
            "description": f"Impute missing values in {len(categorical_missing)} categorical column(s): {', '.join(categorical_missing[:3])}{'...' if len(categorical_missing) > 3 else ''} using most frequent value",
            "affected_columns": categorical_missing,
            "impact": "medium",
            "transformation_type": "handle_missing"
        })
    
    # Strategy 4: Drop rows with missing values (only if missing % is reasonable)
    if eda_stats['missing_overall'] > 0 and eda_stats['missing_overall'] < 30:
        suggestions.append({
            "id": "handle_missing_drop_rows",
            "title": "Remove Rows with Missing Values",
            "description": f"Drop rows containing any missing values (affects {eda_stats['missing_overall']:.1f}% of data)",
            "affected_columns": list(cols_with_missing.keys())[:10],
            "impact": "high" if eda_stats['missing_overall'] > 15 else "medium",
            "transformation_type": "handle_missing"
        })
    
    # Strategy 5: Drop columns with high missing percentage
    high_missing_cols = [
        col for col, stats in eda_stats['columns'].items() 
        if stats['missing_pct'] > 50
    ]
    if high_missing_cols:
        suggestions.append({
            "id": "handle_missing_drop_cols",
            "title": "Remove High-Missing Columns",
            "description": f"Drop {len(high_missing_cols)} column(s) with >50% missing values: {', '.join(high_missing_cols[:3])}{'...' if len(high_missing_cols) > 3 else ''}",
            "affected_columns": high_missing_cols,
            "impact": "high",
            "transformation_type": "handle_missing"
        })
    
    # Remove duplicates (only if duplicates exist)
    if eda_stats['duplicate_rows'] > 0:
        suggestions.append({
            "id": "remove_duplicates_1",
            "title": "Remove Duplicate Rows",
            "description": f"Remove {eda_stats['duplicate_rows']:,} duplicate rows from the dataset",
            "affected_columns": ["all"],
            "impact": "medium",
            "transformation_type": "remove_duplicates"
        })
    
    # Encode categorical variables (only actual categorical columns)
    categorical_cols = [
        col for col, stats in eda_stats['columns'].items() 
        if stats['type'] == 'categorical'
    ]
    if categorical_cols:
        suggestions.append({
            "id": "encode_categorical_1",
            "title": "Encode Categorical Variables",
            "description": f"Convert {len(categorical_cols)} categorical column(s) to numerical: {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''} using label encoding",
            "affected_columns": categorical_cols,
            "impact": "high",
            "transformation_type": "encode_categorical"
        })
    
    # Scale numerical features (only if multiple numerical columns exist)
    numerical_cols = [
        col for col, stats in eda_stats['columns'].items() 
        if stats['type'] == 'numerical'
    ]
    if len(numerical_cols) > 1:
        suggestions.append({
            "id": "scale_numerical_1",
            "title": "Standardize Numerical Features",
            "description": f"Normalize {len(numerical_cols)} numerical column(s): {', '.join(numerical_cols[:3])}{'...' if len(numerical_cols) > 3 else ''} using StandardScaler (mean=0, std=1)",
            "affected_columns": numerical_cols,
            "impact": "medium",
            "transformation_type": "scale_numerical"
        })
    
    return suggestions


def parse_chat_transformation(user_message, df_info, df_head=None):
    """Parse natural language transformation request and generate checkpoints"""
    
    context_str = ""
    if df_head is not None:
        context_str += f"\nDATAFRAME HEAD (First 5 rows):\n{df_head}\n"
    
    prompt = f"""You are a data transformation assistant. The user wants to transform their dataset.

Dataset Info:
- Columns: {', '.join(df_info['columns'])}
- Rows: {df_info['rows']}
{context_str}
User Request: "{user_message}"

Generate a step-by-step transformation plan. Return ONLY a valid JSON object in this EXACT format:
{{
  "title": "Concise 3-5 word title of the transformation",
  "understanding": "Brief summary of what user wants",
  "checkpoints": [
    {{
      "id": "step_1",
      "description": "Detailed step description",
      "action": "specific_action_to_take",
      "status": "pending"
    }}
  ],
  "code_preview": "# Python code snippet"
}}

Return ONLY the JSON object, no other text."""

    response = call_ollama(prompt)
    
    if response:
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
        except json.JSONDecodeError:
            pass
    
    # Fallback
    return {
        "understanding": f"Process request: {user_message}",
        "checkpoints": [
            {
                "id": "step_1",
                "description": "Analyze the request and identify affected columns",
                "action": "analyze",
                "status": "pending"
            },
            {
                "id": "step_2",
                "description": "Apply the transformation",
                "action": "transform",
                "status": "pending"
            },
            {
                "id": "step_3",
                "description": "Validate the results",
                "action": "validate",
                "status": "pending"
            }
        ],
        "code_preview": "# Transformation will be applied based on your request"
    }


def generate_transformation_code(user_request, available_columns, df_head=None, df_dtypes=None):
    """Generate Python code for transformation using LLM"""
    
    context_str = ""
    if df_head is not None:
        context_str += f"\nDATAFRAME HEAD (First 5 rows):\n{df_head}\n"
    
    if df_dtypes is not None:
        context_str += f"\nCOLUMN DATA TYPES:\n{df_dtypes}\n"
    
    prompt = f"""You are a Python data transformation expert. Generate ONLY executable Python code based on the user's request.

AVAILABLE COLUMNS IN DATAFRAME:
{', '.join(available_columns)}
{context_str}
USER REQUEST:
"{user_request}"

CRITICAL RULES:
1. The dataframe is called 'df'
2. Use ONLY columns from the available columns list above
3. Generate ONLY the code to create the new column(s) or transform existing ones
4. Do NOT include any explanations, comments, or markdown
5. Do NOT include import statements (pandas is available as pd, numpy as np)
6. Return ONLY valid Python code that can be executed directly
7. If the user specifies a column name in quotes (like "feature_1_2"), use EXACTLY that name
8. If no specific name is given, create a descriptive name based on the operation
9. You CAN define helper functions if needed, but they must be defined before use
10. Ensure all code is complete and valid syntax

EXAMPLE INPUT: "create a column called \"feature_1_2\" by multiplying feature_1 and feature_2"
EXAMPLE OUTPUT: df['feature_1_2'] = df['feature_1'] * df['feature_2']

EXAMPLE INPUT: "extract year from date_column"
EXAMPLE OUTPUT: df['year'] = pd.to_datetime(df['date_column']).dt.year

Now generate the code for the user's request. Return ONLY the Python code, nothing else:"""

    response = call_ollama(prompt)
    
    if response:
        # Extract code from response (remove any markdown formatting)
        code = response.strip()
        
        # Remove markdown code blocks if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        # Clean up code - remove lines that are clearly not code (like "Here is the code:")
        code_lines = []
        for line in code.split('\n'):
            # Skip lines that look like conversational text
            if line.strip().lower().startswith(('here is', 'sure', 'certainly', 'i have')):
                continue
            code_lines.append(line)
            
        if code_lines:
            return '\n'.join(code_lines)
    
    return "# Error: Could not connect to AI service (Ollama). Please ensure it is running on port 11434."


def apply_transformation(df, transformation_type, params):
    """Apply a specific transformation to the dataframe"""
    import pandas as pd
    import numpy as np
    df_copy = df.copy()
    affected_cols = params.get('affected_columns', [])
    
    try:
        # --- Imputation ---
        if transformation_type == "fillna_mean":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                    
        elif transformation_type == "fillna_median":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                    
        elif transformation_type == "fillna_mode":
            for col in affected_cols:
                if col in df_copy.columns:
                    mode_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Unknown"
                    df_copy[col] = df_copy[col].fillna(mode_val)
                    
        elif transformation_type == "fillna_ffill":
            for col in affected_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].ffill()
                    
        elif transformation_type == "fillna_bfill":
            for col in affected_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].bfill()

        # --- Removal (In-place as it affects rows) ---
        elif transformation_type == "dropna_rows":
            df_copy = df_copy.dropna()
            
        elif transformation_type == "drop_duplicates_all":
            df_copy = df_copy.drop_duplicates()
            
        elif transformation_type == "drop_duplicates_subset":
            if affected_cols:
                df_copy = df_copy.drop_duplicates(subset=affected_cols)

        # --- Scaling ---
        elif transformation_type == "standard_scaler":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    mean_val = df_copy[col].mean()
                    std_val = df_copy[col].std()
                    if std_val != 0:
                        df_copy[col] = (df_copy[col] - mean_val) / std_val
                    else:
                        df_copy[col] = 0
                        
        elif transformation_type == "minmax_scaler":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    min_val = df_copy[col].min()
                    max_val = df_copy[col].max()
                    if max_val != min_val:
                        df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
                    else:
                        df_copy[col] = 0
                        
        elif transformation_type == "robust_scaler":
            scaler = RobustScaler()
            valid_cols = [c for c in affected_cols if c in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[c])]
            if valid_cols:
                df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
                
        elif transformation_type == "maxabs_scaler":
            scaler = MaxAbsScaler()
            valid_cols = [c for c in affected_cols if c in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[c])]
            if valid_cols:
                df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
                
        elif transformation_type == "unit_vector_scaler":
            scaler = Normalizer()
            valid_cols = [c for c in affected_cols if c in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[c])]
            if valid_cols:
                df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])

        # --- Distribution ---
        elif transformation_type == "log_transform":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Ensure positive values
                    if (df_copy[col] <= 0).any():
                        # Shift to positive
                        min_val = df_copy[col].min()
                        df_copy[col] = np.log1p(df_copy[col] - min_val + 1)
                    else:
                        df_copy[col] = np.log1p(df_copy[col])
                        
        elif transformation_type == "sqrt_transform":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Ensure non-negative
                    if (df_copy[col] < 0).any():
                        min_val = df_copy[col].min()
                        df_copy[col] = np.sqrt(df_copy[col] - min_val)
                    else:
                        df_copy[col] = np.sqrt(df_copy[col])
                        
        elif transformation_type == "boxcox_transform":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Box-Cox requires positive data
                    if (df_copy[col] <= 0).any():
                        min_val = df_copy[col].min()
                        data_shifted = df_copy[col] - min_val + 1
                        df_copy[col], _ = stats.boxcox(data_shifted)
                    else:
                        df_copy[col], _ = stats.boxcox(df_copy[col])
                        
        elif transformation_type == "yeojohnson_transform":
            pt = PowerTransformer(method='yeo-johnson')
            valid_cols = [c for c in affected_cols if c in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[c])]
            if valid_cols:
                df_copy[valid_cols] = pt.fit_transform(df_copy[valid_cols])
                
        elif transformation_type == "quantile_transform":
            output_dist = params.get('output_distribution', 'normal')
            qt = QuantileTransformer(output_distribution=output_dist)
            valid_cols = [c for c in affected_cols if c in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[c])]
            if valid_cols:
                df_copy[valid_cols] = qt.fit_transform(df_copy[valid_cols])
                
        elif transformation_type == "reciprocal_transform":
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Avoid division by zero
                    df_copy[col] = 1 / (df_copy[col] + 1e-6)

        # --- Encoding ---
        elif transformation_type == "label_encoder":
            le = LabelEncoder()
            for col in affected_cols:
                if col in df_copy.columns:
                    # Convert to string first to handle mixed types
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    
        elif transformation_type == "onehot_encoder":
            valid_cols = [c for c in affected_cols if c in df_copy.columns]
            if valid_cols:
                df_copy = pd.get_dummies(df_copy, columns=valid_cols, drop_first=True)
                
        elif transformation_type == "ordinal_encoder":
            oe = OrdinalEncoder()
            valid_cols = [c for c in affected_cols if c in df_copy.columns]
            if valid_cols:
                # Ensure string type
                for col in valid_cols:
                    df_copy[col] = df_copy[col].astype(str)
                df_copy[valid_cols] = oe.fit_transform(df_copy[valid_cols])
        
        # --- Outliers ---
        elif transformation_type == "remove_outliers_iqr":
            multiplier = float(params.get('multiplier', 1.5))
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_copy = df_copy[(df_copy[col] >= Q1 - multiplier * IQR) & (df_copy[col] <= Q3 + multiplier * IQR)]
                    
        elif transformation_type == "remove_outliers_zscore":
            threshold = float(params.get('threshold', 3.0))
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    z_scores = np.abs(stats.zscore(df_copy[col].dropna()))
                    df_copy = df_copy[z_scores < threshold]
                    
        elif transformation_type == "clip_outliers":
            lower = float(params.get('lower', 0.01))
            upper = float(params.get('upper', 0.99))
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    lower_bound = df_copy[col].quantile(lower)
                    upper_bound = df_copy[col].quantile(upper)
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                    
        elif transformation_type == "winsorize_outliers":
            limits = float(params.get('limits', 0.05))
            for col in affected_cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col] = stats.mstats.winsorize(df_copy[col], limits=[limits, limits])

        return df_copy, True, "Transformation applied successfully"
    
    except Exception as e:
        print(f"Transformation error: {str(e)}")
        traceback.print_exc()
        return df, False, f"Error: {str(e)}"


def generate_error_resolution(code, error_message, df_info):
    """Generate resolution for transformation error using LLM"""
    
    prompt = f"""You are a Python debugging expert. A data transformation operation failed.
    
CONTEXT:
- Columns: {', '.join(df_info.get('columns', []))}
- Rows: {df_info.get('rows', 0)}

FAILED CODE:
{code}

ERROR MESSAGE:
{error_message}

Analyze the error and provide a user-friendly explanation and a specific resolution.
Return ONLY a valid JSON object in this EXACT format:
{{
  "explanation": "Concise 1-sentence explanation of what went wrong",
  "resolution": "Specific instruction on how to fix it (e.g., 'Ensure column X exists' or 'Convert column Y to numeric first')"
}}"""

    response = call_ollama(prompt)
    
    if response:
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return result
        except json.JSONDecodeError:
            pass
            
    return {
        "explanation": "An error occurred during transformation.",
        "resolution": "Please check the data types and column names."
    }

