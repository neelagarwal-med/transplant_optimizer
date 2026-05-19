import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.utils import resample
from sklearn.frozen import FrozenEstimator
import warnings
warnings.filterwarnings('ignore')

# --- 1. DEPLOYMENT FEATURE ISOLATION ---
def enforce_deployment_features(df: pd.DataFrame) -> pd.DataFrame:
    ui_features = [
        'is_transplanted', 'donor_age', 'donor_bmi', 'terminal_creatinine', 
        'history_htn', 'history_diabetes', 'cause_of_death', 'kdpi', 
        'is_dcd', 'hcv_status'
    ]
    
    available_features = [f for f in ui_features if f in df.columns]
    missing = set(ui_features) - set(available_features)
    
    if missing:
        print(f"⚠️ Note: Missing {missing} from CSV. Continuing with {len(available_features)-1} features.")
        
    return df[available_features].copy()

# --- 2. INGESTION & FORMATTING ---
def clean_and_prepare_optn_data(filepath: str) -> pd.DataFrame:
    print("Loading massive UNOS dataset...")
    df = pd.read_csv(filepath, low_memory=False)
    
    if 'is_transplanted' not in df.columns:
        raise ValueError("Target column 'is_transplanted' not found.")

    df = enforce_deployment_features(df)

    print("Casting types for LightGBM Native Handling...")
    binary_map = {'Y': 1, 'N': 0, 'P': 1, 'U': 0, 'ND': 0, 'Yes': 1, 'No': 0}
    
    for c in df.columns:
        if c == 'is_transplanted': continue
        
        if c in ['donor_age', 'donor_bmi', 'terminal_creatinine', 'kdpi']:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        else:
            if df[c].dtype == 'object' and set(df[c].dropna().unique()).issubset(set(binary_map.keys())):
                df[c] = df[c].map(binary_map).astype(float)
            elif df[c].dtype == 'object':
                df[c] = df[c].astype('category')
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)

    return df

# --- 3. HYPERPARAMETER TUNING & TRAINING ---
def train_tuned_calibrated_pipeline(X: pd.DataFrame, y: pd.Series):
    print(f"\n🧠 Initiating Hyperparameter Optimization on {X.shape[0]} samples...")
    print(f"Features mapped and secured: {X.columns.tolist()}")
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=42)
    
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05],
        'num_leaves': [15, 31, 63],
        'max_depth': [4, 6, 8],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # FIX 1: verbose=-1 silences the "No further splits" warning
    base_estimator = lgb.LGBMClassifier(
        n_estimators=300, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        verbose=-1 
    )
    
    # FIX 2: n_jobs=1 prevents the MacBook multiprocessing LokyProcess freeze
    rs = RandomizedSearchCV(
        estimator=base_estimator, param_distributions=param_grid, 
        n_iter=10, scoring='average_precision', cv=3, random_state=42, n_jobs=1
    )
    
    print("Searching for optimal tree parameters (This may take ~2 minutes. Let it run!)...")
    rs.fit(
        X_train, y_train, 
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)], 
        eval_set=[(X_val, y_val)]
    )
    best_lgbm = rs.best_estimator_

    print("Applying Isotonic Calibration...")
    # Wrap best_lgbm in FrozenEstimator to lock its weights
    # Remove cv=3 so it doesn't try to cross-validate and refit
    calibrated_clf = CalibratedClassifierCV(
        estimator=FrozenEstimator(best_lgbm), 
        method='isotonic'
    )
    calibrated_clf.fit(X_val, y_val)
    
    evaluate_with_confidence_intervals(calibrated_clf, X_test, y_test)
    
    return calibrated_clf, best_lgbm

# --- 4. STATISTICAL VALIDATION (BOOTSTRAPPING) ---
def evaluate_with_confidence_intervals(model, X_test, y_test, n_bootstraps=1000):
    print("\n--- Final Test Set Clinical Metrics (95% CI) ---")
    preds_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, preds_proba)
    pr_auc = average_precision_score(y_test, preds_proba)
    brier = brier_score_loss(y_test, preds_proba)
    
    roc_aucs, pr_aucs = [], []
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(preds_proba)))
        if len(np.unique(y_test.iloc[indices])) < 2: continue
        roc_aucs.append(roc_auc_score(y_test.iloc[indices], preds_proba[indices]))
        pr_aucs.append(average_precision_score(y_test.iloc[indices], preds_proba[indices]))
        
    roc_ci = np.percentile(roc_aucs, [2.5, 97.5])
    pr_ci = np.percentile(pr_aucs, [2.5, 97.5])
    
    print(f"ROC-AUC: {roc_auc:.3f} (95% CI: {roc_ci[0]:.3f} - {roc_ci[1]:.3f})")
    print(f"PR-AUC:  {pr_auc:.3f} (95% CI: {pr_ci[0]:.3f} - {pr_ci[1]:.3f})")
    print(f"Brier:   {brier:.3f}")
    print("------------------------------------------------")

# --- 5. SHAP EXPLAINABILITY ---
def predict_with_explanations(calibrated_clf, base_lgbm, patient_data: pd.DataFrame):
    prob_success = calibrated_clf.predict_proba(patient_data)[0, 1]
    
    explainer = shap.TreeExplainer(base_lgbm)
    shap_values = explainer.shap_values(patient_data)
    shap_vals_target = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
    feature_importance = pd.DataFrame({
        'Feature': patient_data.columns.tolist(),
        'Impact': shap_vals_target,
        'Value': patient_data.iloc[0].values
    })
    
    top_3 = feature_importance.reindex(feature_importance['Impact'].abs().sort_values(ascending=False).index).head(3)
    
    drivers_formatted = [
        f"{row['Feature'].replace('_', ' ').title()} ({row['Value']}) {'Increased' if row['Impact'] > 0 else 'Decreased'} utilization probability."
        for _, row in top_3.iterrows()
    ]
    return prob_success, drivers_formatted

# --- EXECUTION & ARTIFACT EXPORT ---
if __name__ == "__main__":
    df_clean = clean_and_prepare_optn_data('master_kidney_dataset_ALL_COLS.csv')
    
    X = df_clean.drop(columns=['is_transplanted'])
    y = df_clean['is_transplanted']
    
    calibrated_model, base_model = train_tuned_calibrated_pipeline(X, y)
    
    print("\nSaving deployment artifacts...")
    deployment_package = {
        'calibrator': calibrated_model,
        'base_model': base_model,
        'features': X.columns.tolist() 
    }
    joblib.dump(deployment_package, 'kidney_utilization_lgbm.pkl')
    print("✅ System Ready. Boot up Streamlit to view.")