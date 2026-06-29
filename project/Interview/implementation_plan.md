# Production Feature Selection Pipeline

This plan outlines the implementation of an end-to-end, production-grade feature selection pipeline for the Customer Churn model, strictly following the 8 steps provided in the prompt.

*Note: As requested, we will skip data aggregation/merging and apply this pipeline directly to your already-created master feature table.*

## User Review Required

> [!IMPORTANT]
> The pipeline requires training multiple models (Logistic Regression, Random Forest, XGBoost) and calculating SHAP values. This will be computationally heavy. Please approve this plan to proceed. 

## Open Questions
> [!WARNING]
> Please confirm where your master table is currently stored. Is it saved as a CSV file (e.g., `final_feature_table.csv`) or is it currently just a variable loaded in memory inside your notebook? (If it's just in memory, I will generate the Python code for you to paste and run directly in your notebook).

## Proposed Execution

I will generate a robust Python script (or notebook cell block) that takes your `master_df` as input and runs the following strictly defined steps:

1. **Step 1: Business Filtering:** Identify and drop pure ID columns and irrelevant metadata. (Outputs status table).
2. **Step 2: Leakage Detection:** Drop `churn_reason`, `predicted_churn_prob`, `retention_offer_sent`, `offer_accepted` and generate a leakage report.
3. **Step 3: Data Quality:** Filter columns with >60% nulls and zero variance. (Outputs quality report).
4. **Step 4: Statistical Filtering:** Calculate Mutual Information and a Correlation matrix to drop heavily correlated pairs (>0.90).
5. **Step 5: Model-Based Selection:** Train Random Forest, XGBoost, and Logistic Regression on the remaining features to extract consistent feature importances.
6. **Step 6: SHAP Analysis:** Run `shap.TreeExplainer` on XGBoost to get the global SHAP values and identify top churn-drivers.
7. **Step 7: Stability Testing:** Perform a simulated temporal split to detect unstable features.
8. **Step 8: Cost/Latency Filtering:** Categorize remaining features by complexity and filter out low-ROI, high-cost features.

### Final Outputs Generated

At the end of execution, you will receive:
1. `FINAL_FEATURES` (List)
2. `REMOVED_FEATURES` (Categorized JSON Dict)
3. **Production Summary Report** containing all the metrics, reports, and tables you requested.

## Verification Plan
- I will use `random_state=42` across all algorithms to ensure exact reproducibility.
- I will explicitly document the `Reason` for every single dropped feature as required.
