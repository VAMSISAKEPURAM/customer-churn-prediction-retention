import pandas as pd
import numpy as np
import scipy.stats as ss
import hashlib
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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

class ProductionEncoder:
    def __init__(self, target_col='churned'):
        self.target_col = target_col
        self.encoding_plan = {}
        self.encoders = {}
        self.target_means = {}
        self.global_mean = 0
        self.freq_maps = {}
        self.ordinal_cols = ['rfm_segment', 'engagement_tier']
        self.ordinal_categories = {
            'rfm_segment': ['Lost', 'At Risk', 'New', 'Promising', 'Loyal', 'Champions'],
            'engagement_tier': ['dormant', 'low', 'medium', 'high']
        }
        self.hash_bins = 100
        
    def fit_transform(self, df_train, df_test):
        df_train = df_train.copy()
        df_test = df_test.copy()
        
        self.global_mean = df_train[self.target_col].mean()
        cat_cols = df_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)
            
        for col in cat_cols:
            df_train[col] = df_train[col].astype(str).replace('nan', 'Unknown')
            df_test[col] = df_test[col].astype(str).replace('nan', 'Unknown')
            
            unique_vals = df_train[col].nunique()
            
            # Check for strong correlation (prevent leakage by ensuring enough samples per category)
            # Only consider correlation if avg samples per category > 5
            cv = 0
            if len(df_train) / unique_vals > 5 and unique_vals > 1:
                cv = cramers_v(df_train[col], df_train[self.target_col])
                
            if col in self.ordinal_cols:
                self.encoding_plan[col] = 'Ordinal'
                enc = OrdinalEncoder(categories=[self.ordinal_categories[col]], handle_unknown='use_encoded_value', unknown_value=-1)
                df_train[f"{col}_ord"] = enc.fit_transform(df_train[[col]])
                df_test[f"{col}_ord"] = enc.transform(df_test[[col]])
                self.encoders[col] = enc
                df_train.drop(columns=[col], inplace=True)
                df_test.drop(columns=[col], inplace=True)
                
            elif cv > 0.15:
                self.encoding_plan[col] = 'Target'
                # Fit target encoding only on train
                means = df_train.groupby(col)[self.target_col].mean().to_dict()
                self.target_means[col] = means
                df_train[f"{col}_te"] = df_train[col].map(means).fillna(self.global_mean)
                df_test[f"{col}_te"] = df_test[col].map(means).fillna(self.global_mean)
                df_train.drop(columns=[col], inplace=True)
                df_test.drop(columns=[col], inplace=True)
                
            elif unique_vals <= 10:
                self.encoding_plan[col] = 'One-Hot'
                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                train_encoded = enc.fit_transform(df_train[[col]])
                test_encoded = enc.transform(df_test[[col]])
                col_names = [f"{col}_ohe_{cat}" for cat in enc.categories_[0]]
                
                train_encoded_df = pd.DataFrame(train_encoded, columns=col_names, index=df_train.index)
                test_encoded_df = pd.DataFrame(test_encoded, columns=col_names, index=df_test.index)
                
                df_train = pd.concat([df_train, train_encoded_df], axis=1)
                df_test = pd.concat([df_test, test_encoded_df], axis=1)
                self.encoders[col] = enc
                df_train.drop(columns=[col], inplace=True)
                df_test.drop(columns=[col], inplace=True)
                
            elif unique_vals <= 50:
                self.encoding_plan[col] = 'Frequency'
                freq = df_train[col].value_counts(normalize=True).to_dict()
                self.freq_maps[col] = freq
                df_train[f"{col}_freq"] = df_train[col].map(freq).fillna(0)
                df_test[f"{col}_freq"] = df_test[col].map(freq).fillna(0)
                df_train.drop(columns=[col], inplace=True)
                df_test.drop(columns=[col], inplace=True)
                
            else:
                self.encoding_plan[col] = 'Hash'
                df_train[f"{col}_hash"] = df_train[col].apply(lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % self.hash_bins)
                df_test[f"{col}_hash"] = df_test[col].apply(lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % self.hash_bins)
                df_train.drop(columns=[col], inplace=True)
                df_test.drop(columns=[col], inplace=True)
                
        return df_train, df_test

def main():
    print("Loading dataset...")
    df = pd.read_csv('Data/final_customer_feature_table.csv')
    target = 'churned' if 'churned' in df.columns else 'churn'
    
    print("Splitting train/test...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    
    print("Encoding...")
    encoder = ProductionEncoder(target_col=target)
    df_train_enc, df_test_enc = encoder.fit_transform(df_train, df_test)
    
    # Save artifacts
    print("Saving artifacts...")
    os.makedirs('artifacts', exist_ok=True)
    artifacts = {
        'encoding_plan': encoder.encoding_plan,
        'sklearn_encoders': encoder.encoders,
        'target_means': encoder.target_means,
        'global_mean': encoder.global_mean,
        'freq_maps': encoder.freq_maps,
        'hash_bins': encoder.hash_bins
    }
    joblib.dump(artifacts, 'artifacts/categorical_encoders.pkl')
    
    with open('artifacts/encoding_plan.json', 'w') as f:
        json.dump(encoder.encoding_plan, f, indent=4)
        
    df_train_enc.to_csv('Data/train_encoded.csv', index=False)
    df_test_enc.to_csv('Data/test_encoded.csv', index=False)
    
    print("="*50)
    print("ENCODING_PLAN = {")
    for k, v in encoder.encoding_plan.items():
        print(f'    "{k}": "{v}",')
    print("}")
    print("="*50)
    
    print(f"1. Final encoded dataframe shape: Train: {df_train_enc.shape}, Test: {df_test_enc.shape}")
    
    new_cols = [c for c in df_train_enc.columns if c not in df.columns]
    print(f"2. Newly created columns ({len(new_cols)}): {new_cols[:10]} ...")
    
    print(f"3. Saved encoder/mapping artifacts: artifacts/categorical_encoders.pkl, artifacts/encoding_plan.json")
    
if __name__ == '__main__':
    main()
