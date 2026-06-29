import pandas as pd
import numpy as np
import json
import scipy.stats as ss

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Load data (only a sample or specific columns for speed, but let's read the full data as we need correlations)
df = pd.read_csv('Data/final_customer_feature_table.csv')
target = 'churned' if 'churned' in df.columns else 'churn'

# Find categorical columns
cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

if target in cat_cols:
    cat_cols.remove(target)

# We should also exclude likely IDs or date columns if they are strings, but user says "detect all categorical columns".
# Let's check for date columns and IDs heuristically, though we might still just report them.
# 'customer_id', 'billing_id', 'metric_id'

summary = {}
for col in cat_cols:
    unique_vals = df[col].dropna().unique().tolist()
    cardinality = len(unique_vals)
    
    # Calculate Cramer's V for correlation with target
    # Handle NaNs by dropping for correlation calculation
    valid_idx = df[col].notna() & df[target].notna()
    if valid_idx.sum() > 0:
        cv = cramers_v(df.loc[valid_idx, col], df.loc[valid_idx, target])
    else:
        cv = 0.0

    summary[col] = {
        'cardinality': cardinality,
        'unique_values': unique_vals[:20],
        'cramers_v': cv
    }

with open('categorical_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
