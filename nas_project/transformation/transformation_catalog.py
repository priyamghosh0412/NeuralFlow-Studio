# Transformation Catalog
# Defines all available hardcoded transformation cards

TRANSFORMATION_CATALOG = {
    "missing_values": {
        "title": "Missing Values",
        "cards": [
            {
                "id": "fillna_mean",
                "title": "Fill Missing with Mean",
                "impact": "Medium",
                "description": "Replaces missing values with the mean of the column. Preserves average but sensitive to outliers. Best for normally distributed data.",
                "type": "imputation",
                "function": "fillna_mean",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "fillna_median",
                "title": "Fill Missing with Median",
                "impact": "Medium",
                "description": "Replaces missing values with the median of the column. Robust to outliers and preserves distribution. Best for skewed data.",
                "type": "imputation",
                "function": "fillna_median",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "fillna_mode",
                "title": "Fill Missing with Mode",
                "impact": "Low",
                "description": "Replaces missing values with the most frequent value. Best for categorical features or discrete numerical data.",
                "type": "imputation",
                "function": "fillna_mode",
                "requires_numeric": False,
                "params": {}
            },
            {
                "id": "fillna_ffill",
                "title": "Forward Fill Missing",
                "impact": "Low",
                "description": "Fills missing values with the previous valid value. Best for time-series data with temporal continuity.",
                "type": "imputation",
                "function": "fillna_ffill",
                "requires_numeric": False,
                "params": {}
            },
            {
                "id": "fillna_bfill",
                "title": "Backward Fill Missing",
                "impact": "Low",
                "description": "Fills missing values with the next valid value. Useful for time-series when future values are known.",
                "type": "imputation",
                "function": "fillna_bfill",
                "requires_numeric": False,
                "params": {}
            },
            {
                "id": "dropna_rows",
                "title": "Remove Rows with Missing",
                "impact": "High",
                "description": "Removes all rows containing missing values. May result in significant data loss. Use when missing data is minimal.",
                "type": "removal",
                "function": "dropna_rows",
                "requires_numeric": False,
                "params": {}
            }
        ]
    },
    "duplicates": {
        "title": "Duplicates",
        "cards": [
            {
                "id": "drop_duplicates_all",
                "title": "Remove Duplicate Rows",
                "impact": "Medium",
                "description": "Removes exact duplicate rows keeping first occurrence. Reduces dataset size and prevents bias from repeated samples.",
                "type": "removal",
                "function": "drop_duplicates_all",
                "requires_numeric": False,
                "params": {}
            },
            {
                "id": "drop_duplicates_subset",
                "title": "Remove Duplicates by Columns",
                "impact": "Medium",
                "description": "Removes rows with duplicate values in selected columns. Useful for identifying unique records by key columns.",
                "type": "removal",
                "function": "drop_duplicates_subset",
                "requires_numeric": False,
                "params": {}
            }
        ]
    },
    "outliers": {
        "title": "Outlier Handling",
        "cards": [
            {
                "id": "remove_outliers_iqr",
                "title": "Remove Outliers (IQR)",
                "impact": "High",
                "description": "Removes values beyond 1.5 × IQR from quartiles. Standard statistical method, may remove valid extreme values.",
                "type": "outlier",
                "function": "remove_outliers_iqr",
                "requires_numeric": True,
                "params": {"multiplier": {"type": "number", "default": 1.5, "label": "IQR Multiplier"}}
            },
            {
                "id": "remove_outliers_zscore",
                "title": "Remove Outliers (Z-Score)",
                "impact": "High",
                "description": "Removes values more than 3 standard deviations from mean. Assumes normal distribution.",
                "type": "outlier",
                "function": "remove_outliers_zscore",
                "requires_numeric": True,
                "params": {"threshold": {"type": "number", "default": 3.0, "label": "Z-Score Threshold"}}
            },
            {
                "id": "clip_outliers",
                "title": "Clip Outliers to Percentiles",
                "impact": "Medium",
                "description": "Caps extreme values at percentile thresholds. Preserves all rows while reducing outlier impact.",
                "type": "outlier",
                "function": "clip_outliers",
                "requires_numeric": True,
                "params": {"lower": {"type": "number", "default": 0.01, "label": "Lower Percentile"}, "upper": {"type": "number", "default": 0.99, "label": "Upper Percentile"}}
            },
            {
                "id": "log_transform_outliers",
                "title": "Log Transform Outliers",
                "impact": "Medium",
                "description": "Applies log transformation to compress large values. Effective for right-skewed data with outliers.",
                "type": "transformation",
                "function": "log_transform",
                "requires_numeric": True,
                "requires_positive": True,
                "params": {}
            },
            {
                "id": "winsorize_outliers",
                "title": "Winsorize Outliers",
                "impact": "Medium",
                "description": "Replaces extreme values with percentile boundaries. Less aggressive than removal, preserves sample size.",
                "type": "outlier",
                "function": "winsorize_outliers",
                "requires_numeric": True,
                "params": {"limits": {"type": "number", "default": 0.05, "label": "Limit"}}
            }
        ]
    },
    "scaling": {
        "title": "Scaling & Normalization",
        "cards": [
            {
                "id": "standard_scaler",
                "title": "Standard Scaling (Z-Score)",
                "impact": "Medium",
                "description": "Scales features to zero mean and unit variance. Best for algorithms sensitive to feature scale (SVM, Neural Networks).",
                "type": "scaling",
                "function": "standard_scaler",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "minmax_scaler",
                "title": "Min-Max Normalization",
                "impact": "Medium",
                "description": "Scales features to [0, 1] range. Preserves relationships, sensitive to outliers. Good for bounded algorithms.",
                "type": "scaling",
                "function": "minmax_scaler",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "robust_scaler",
                "title": "Robust Scaling",
                "impact": "Medium",
                "description": "Uses median and IQR for scaling. Robust to outliers. Best when data contains extreme values.",
                "type": "scaling",
                "function": "robust_scaler",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "maxabs_scaler",
                "title": "Max Abs Scaling",
                "impact": "Low",
                "description": "Scales by dividing by maximum absolute value. Preserves sparsity, good for sparse data.",
                "type": "scaling",
                "function": "maxabs_scaler",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "unit_vector_scaler",
                "title": "Unit Vector Normalization",
                "impact": "Low",
                "description": "Scales each sample to unit norm. Useful for text data and when direction matters more than magnitude.",
                "type": "scaling",
                "function": "unit_vector_scaler",
                "requires_numeric": True,
                "params": {}
            }
        ]
    },
    "distribution": {
        "title": "Distribution Transformation",
        "cards": [
            {
                "id": "log_transform",
                "title": "Log Transform",
                "impact": "Medium",
                "description": "Applies log(1+x) transformation. Reduces right skew, makes distribution more normal. Best for positive skewed data.",
                "type": "transformation",
                "function": "log_transform",
                "requires_numeric": True,
                "requires_positive": True,
                "params": {}
            },
            {
                "id": "sqrt_transform",
                "title": "Square Root Transform",
                "impact": "Low",
                "description": "Applies square root transformation. Less aggressive than log, good for moderately skewed data.",
                "type": "transformation",
                "function": "sqrt_transform",
                "requires_numeric": True,
                "requires_non_negative": True,
                "params": {}
            },
            {
                "id": "boxcox_transform",
                "title": "Box-Cox Transform",
                "impact": "High",
                "description": "Finds optimal power transformation to achieve normality. Most powerful but requires positive values.",
                "type": "transformation",
                "function": "boxcox_transform",
                "requires_numeric": True,
                "requires_positive": True,
                "params": {}
            },
            {
                "id": "yeojohnson_transform",
                "title": "Yeo-Johnson Transform",
                "impact": "High",
                "description": "Similar to Box-Cox but works with negative values. Automatically finds best transformation.",
                "type": "transformation",
                "function": "yeojohnson_transform",
                "requires_numeric": True,
                "params": {}
            },
            {
                "id": "quantile_transform",
                "title": "Quantile Transform",
                "impact": "High",
                "description": "Maps values to uniform or normal distribution. Non-linear, preserves rank order. Very effective for skewed data.",
                "type": "transformation",
                "function": "quantile_transform",
                "requires_numeric": True,
                "params": {"output_distribution": {"type": "select", "options": ["normal", "uniform"], "default": "normal", "label": "Output Distribution"}}
            },
            {
                "id": "reciprocal_transform",
                "title": "Reciprocal Transform",
                "impact": "Medium",
                "description": "Applies 1/x transformation. Useful for left-skewed distributions and rate data.",
                "type": "transformation",
                "function": "reciprocal_transform",
                "requires_numeric": True,
                "requires_non_zero": True,
                "params": {}
            }
        ]
    },
    "encoding": {
        "title": "Categorical Encoding",
        "cards": [
            {
                "id": "label_encoder",
                "title": "Label Encoding",
                "impact": "Low",
                "description": "Converts categories to integers (0, 1, 2, ...). Simple but implies ordinal relationship. Use for tree-based models.",
                "type": "encoding",
                "function": "label_encoder",
                "requires_categorical": True,
                "params": {}
            },
            {
                "id": "onehot_encoder",
                "title": "One-Hot Encoding",
                "impact": "Medium",
                "description": "Creates binary column for each category. No ordinal assumption. Increases dimensionality. Best for linear models.",
                "type": "encoding",
                "function": "onehot_encoder",
                "requires_categorical": True,
                "params": {}
            },
            {
                "id": "ordinal_encoder",
                "title": "Ordinal Encoding",
                "impact": "Low",
                "description": "Encodes categories with specified order. Use when categories have natural ordering (low, medium, high).",
                "type": "encoding",
                "function": "ordinal_encoder",
                "requires_categorical": True,
                "params": {}
            }
        ]
    }
}
