import pandas as pd

print("Loading final_df.csv...")
file_path = r'D:\D_Drive\vamsi vs code\EDUKRON_FILES\Customer_churn_prediction_retenction\customer-churn-prediction-retention-model\Data\final_df.csv'

# Load the dataframe
df = pd.read_csv(file_path)

# Find categorical columns
cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

print(f"\nCategorical Columns found: {len(cat_cols)}")
print("="*50)

for col in cat_cols:
    # Get unique values, sort them if possible for cleaner output
    unique_vals = df[col].dropna().unique().tolist()
    
    print(f"Column: {col} | {len(unique_vals)} unique values")
    
    if len(unique_vals) <= 20:
        print(f"Values: {unique_vals}")
    else:
        print(f"Values: {unique_vals[:15]} ... (truncated)")
    print("-" * 50)
