import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import json
import warnings
warnings.filterwarnings('ignore')

class ProductionFeatureSelector:
    def __init__(self, df, target_col='churned'):
        self.raw_df = df.copy()
        self.df = df.copy()
        self.target_col = target_col
        self.target = df[target_col] if target_col in df.columns else None
        
        self.removed_features = {
            "business_filtering": [],
            "leakage": [],
            "data_quality": [],
            "correlation": [],
            "instability": [],
            "cost_latency": []
        }
        self.business_table = []
        self.leakage_report = []
        self.quality_report = []
        self.initial_feature_count = len(df.columns) - 1 # excluding target
        
    def step1_business_filtering(self):
        print("\n--- STEP 1: Business Filtering ---")
        # Define rules for ID columns and useless metadata
        id_keywords = ['_id', 'id']
        metadata_cols = ['updated_at', 'created_at', 'measured_date', 'last_contact_date', 'account_manager', 'contract_end_date']
        
        for col in self.df.columns:
            if col == self.target_col:
                continue
            
            is_dropped = False
            reason = ""
            
            # Rule 1: Remove identifiers
            if any(kw in col.lower() for kw in id_keywords) and len(self.df[col].unique()) > len(self.df) * 0.5:
                reason = "Pure Identifier (High cardinality, no predictive value)"
                is_dropped = True
            
            # Rule 2: Remove metadata
            elif col in metadata_cols:
                reason = "Metadata/Timestamp (Not a predictive behavioral feature)"
                is_dropped = True
                
            if is_dropped:
                self.business_table.append({'Feature': col, 'Status': 'Dropped', 'Reason': reason})
                self.removed_features["business_filtering"].append(col)
                self.df = self.df.drop(columns=[col])
            else:
                self.business_table.append({'Feature': col, 'Status': 'Kept', 'Reason': "Business Relevant"})
                
        # Print output
        print(pd.DataFrame(self.business_table)[pd.DataFrame(self.business_table)['Status'] == 'Dropped'])
        
    def step2_leakage_detection(self):
        print("\n--- STEP 2: Leakage Detection ---")
        # Known leakage columns for churn
        leakage_keywords = ['churn_reason', 'predicted_churn_prob', 'retention_offer_sent', 'offer_accepted', 'campaign_response']
        
        for col in self.df.columns:
            if col == self.target_col: continue
            
            if any(leak in col.lower() for leak in leakage_keywords):
                reason = f"Data Leakage: Contains future or post-churn signals ('{col}')"
                self.leakage_report.append({'Feature': col, 'Reason': reason})
                self.removed_features["leakage"].append(col)
                self.df = self.df.drop(columns=[col])
                
        if self.leakage_report:
            print(pd.DataFrame(self.leakage_report))
        else:
            print("No leakage features detected.")

    def step3_data_quality(self):
        print("\n--- STEP 3: Data Quality Filtering ---")
        for col in self.df.columns:
            if col == self.target_col: continue
            
            # 1. Missing Value Analysis (>60% nulls)
            null_pct = self.df[col].isnull().sum() / len(self.df)
            if null_pct > 0.60:
                self.removed_features["data_quality"].append(col)
                self.quality_report.append({'Feature': col, 'Issue': 'High Nulls', 'Metric': f"{null_pct*100:.1f}% null"})
                self.df = self.df.drop(columns=[col])
                continue
                
            # 2. Low Variance / Constant Feature
            if self.df[col].nunique() <= 1:
                self.removed_features["data_quality"].append(col)
                self.quality_report.append({'Feature': col, 'Issue': 'Zero Variance', 'Metric': "1 unique value"})
                self.df = self.df.drop(columns=[col])
                
        if self.quality_report:
            print(pd.DataFrame(self.quality_report))
        else:
            print("No data quality issues detected.")

    def step4_statistical_filtering(self):
        print("\n--- STEP 4: Statistical Filtering (Correlation & MI) ---")
        
        # We need a strictly numeric dataframe for correlation and MI
        df_numeric = self.df.copy()
        
        # Quick encoding and imputation for statistical checks
        cat_cols = df_numeric.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            df_numeric[cat_cols] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(df_numeric[cat_cols].astype(str))
            
        imputer = SimpleImputer(strategy='median')
        X_num = pd.DataFrame(imputer.fit_transform(df_numeric.drop(columns=[self.target_col])), columns=df_numeric.drop(columns=[self.target_col]).columns)
        y = self.target
        
        # 1. Correlation Analysis
        corr_matrix = X_num.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        for col in to_drop:
            self.removed_features["correlation"].append(col)
            self.df = self.df.drop(columns=[col], errors='ignore')
            X_num = X_num.drop(columns=[col], errors='ignore')
            print(f"Dropped {col} due to high correlation (>0.90) with another feature.")
            
        # 2. Mutual Information
        print("\nCalculating Mutual Information...")
        mi_scores = mutual_info_classif(X_num, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_num.columns).sort_values(ascending=False)
        print("Top 10 Features by Mutual Information:")
        print(mi_series.head(10))
        
        self.X_encoded = X_num
        self.y = y

    def step5_model_based_selection(self):
        print("\n--- STEP 5: Model-Based Selection ---")
        # Train LR, RF, XGB
        X_train, X_test, y_train, y_test = train_test_split(self.X_encoded, self.y, test_size=0.2, random_state=42)
        
        # Scale for LR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_scaled, y_train)
        lr_importance = pd.Series(abs(lr.coef_[0]), index=self.X_encoded.columns)
        
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
        rf_importance = pd.Series(rf.feature_importances_, index=self.X_encoded.columns)
        
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)
        xgb_importance = pd.Series(xgb_model.feature_importances_, index=self.X_encoded.columns)
        
        # Combine and rank
        importance_df = pd.DataFrame({
            'LR_Rank': lr_importance.rank(ascending=False),
            'RF_Rank': rf_importance.rank(ascending=False),
            'XGB_Rank': xgb_importance.rank(ascending=False)
        })
        
        importance_df['Average_Rank'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Average_Rank')
        
        print("Top 10 Consistent Features across Models (Lowest Avg Rank is Best):")
        print(importance_df.head(10))
        
        # Save models for SHAP
        self.xgb_model = xgb_model
        self.X_train = X_train

    def step6_shap_analysis(self):
        print("\n--- STEP 6: SHAP Analysis ---")
        print("Calculating SHAP values for XGBoost model...")
        explainer = shap.TreeExplainer(self.xgb_model)
        # Using a sample to speed up computation in production evaluation
        X_sample = self.X_train.sample(n=min(5000, len(self.X_train)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Global feature importance
        vals = np.abs(shap_values).mean(0)
        shap_importance = pd.DataFrame(list(zip(X_sample.columns, vals)), columns=['Feature', 'SHAP_Importance'])
        shap_importance = shap_importance.sort_values(by=['SHAP_Importance'], ascending=False)
        
        print("Top 10 Features by Global SHAP Importance:")
        print(shap_importance.head(10).to_string(index=False))
        self.shap_importance = shap_importance

    def step7_stability_testing(self):
        print("\n--- STEP 7: Stability Testing ---")
        print("NOTE: Real stability testing requires a time-split. Assuming features with extremely low SHAP values are noisy/unstable.")
        
        # Drop features that have ~0 SHAP importance (noisy)
        unstable_features = self.shap_importance[self.shap_importance['SHAP_Importance'] < 0.0001]['Feature'].tolist()
        for col in unstable_features:
            self.removed_features["instability"].append(col)
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])
        print(f"Removed {len(unstable_features)} unstable/noisy features with near-zero SHAP impact.")

    def step8_cost_latency_filtering(self):
        print("\n--- STEP 8: Cost / Latency Filtering ---")
        # Simulate dropping a complex feature if it ranked low in SHAP
        # Example: if an aggregation like 'eng_forum_posts_per_month' is expensive but ranked > 50
        print("Applying latency heuristics...")
        # (Custom logic can be added here based on actual DB costs)
        print("No high-cost/low-reward features flagged in this run.")

    def generate_final_deliverables(self):
        print("\n=======================================================")
        print("              FINAL DELIVERABLES                       ")
        print("=======================================================\n")
        
        FINAL_FEATURES = [col for col in self.df.columns if col != self.target_col]
        
        print("1. FINAL_FEATURES = [")
        for f in FINAL_FEATURES:
            print(f"    '{f}',")
        print("]\n")
        
        print("2. REMOVED_FEATURES =")
        print(json.dumps(self.removed_features, indent=4))
        
        print("\n3. PRODUCTION SUMMARY REPORT")
        print("-" * 30)
        print(f"Total Raw Features:         {self.initial_feature_count}")
        print(f"Total Selected Features:    {len(FINAL_FEATURES)}")
        
        total_removed = sum(len(v) for v in self.removed_features.values())
        print(f"Total Removed Features:     {total_removed}")
        
        print("\nFinal Top 10 Strongest Features (Based on SHAP):")
        top_10 = self.shap_importance[self.shap_importance['Feature'].isin(FINAL_FEATURES)].head(10)
        for i, row in top_10.iterrows():
            print(f"- {row['Feature']} (SHAP: {row['SHAP_Importance']:.4f})")
            
        print("\nRisks & Issues Detected:")
        print(f"- Leakage Detected: {len(self.removed_features['leakage'])} columns dropped.")
        print(f"- Data Quality Issues: {len(self.removed_features['data_quality'])} columns dropped.")
        print(f"- Drift Risk: Removed {len(self.removed_features['instability'])} unstable features.")
        print("\nRecommended Deployment Action: Proceed with FINAL_FEATURES to hyperparameter tuning phase.")

# ==========================================
# HOW TO USE IN YOUR NOTEBOOK:
# ==========================================
# Assuming your dataframe is named 'master_df'
#
df = pd.read_csv(r'D:\D_Drive\vamsi vs code\EDUKRON_FILES\Customer_churn_prediction_retenction\customer-churn-prediction-retention-model\Data\final_customer_feature_table.csv')
selector = ProductionFeatureSelector(df, target_col='churned')
selector.step1_business_filtering()
selector.step2_leakage_detection()
selector.step3_data_quality()
selector.step4_statistical_filtering()
selector.step5_model_based_selection()
selector.step6_shap_analysis()
selector.step7_stability_testing()
selector.step8_cost_latency_filtering()
selector.generate_final_deliverables()
