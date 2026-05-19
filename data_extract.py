import pandas as pd
import csv
import re
from io import StringIO

def get_all_col_names(htm_file):
    """Bulletproof parser: Scans the HTML to find the true UNOS Variable names."""
    with open(htm_file, 'r', encoding='windows-1252') as f:
        html_content = f.read()
    
    # UNOS dictionaries often have multiple tables; grab the main one
    tables = pd.read_html(StringIO(html_content))
    df_layout = max(tables, key=lambda df: df.shape[0])
    
    # Content-Aware Hunt: Find the column that actually contains UNOS codes
    for col in df_layout.columns:
        col_values = df_layout[col].astype(str).str.strip().str.upper().tolist()
        if any('AGE_DON' in val or 'DON_AGE' in val for val in col_values):
            return col_values
            
    # Absolute fallback to your original working logic
    if 'LABEL' in df_layout.columns:
        return df_layout['LABEL'].astype(str).str.strip().str.upper().tolist()
    return df_layout.iloc[:, 1].astype(str).str.strip().str.upper().tolist()

def build_final_dataset(donor_dat, donor_htm, output_file):
    print("🚀 Building Master Transplant Dataset (Donor-Only Architecture)...")

    # 1. EXTRACT AND LOAD DONOR DATA
    try:
        donor_all_cols = get_all_col_names(donor_htm)
        print(f"Loading ALL {len(donor_all_cols)} Donor columns...")
        df_donor = pd.read_csv(
            donor_dat, sep='\t', header=None, names=donor_all_cols, 
            encoding='latin-1', quoting=csv.QUOTE_NONE, dtype=str, low_memory=False
        )
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 2. BULLETPROOF FEATURE MAPPING
    print("Hunting for clinical variables...")
    final_rename = {}
    for col in df_donor.columns:
        # Strip invisible SAS spaces
        c_up = re.sub(r'[^A-Z0-9_]', '', str(col).upper())
        
        if c_up in ['AGE_DON', 'DON_AGE']: final_rename[col] = 'donor_age'
        elif c_up in ['BMI_DON_CALC', 'DON_BMI']: final_rename[col] = 'donor_bmi'
        elif c_up in ['CREAT_DON', 'DON_CREAT']: final_rename[col] = 'terminal_creatinine'
        elif c_up in ['HIST_HYPERTENS_DON', 'DON_HIST_HYPERTEN']: final_rename[col] = 'history_htn'
        elif c_up in ['HIST_DIABETES_DON', 'DIABETES_DON', 'DON_HIST_DIAB']: final_rename[col] = 'history_diabetes'
        elif c_up in ['COD_CAD_DON', 'DON_CAD_DON_COD']: final_rename[col] = 'cause_of_death'
        elif c_up == 'KDPI': final_rename[col] = 'kdpi'
        elif c_up in ['NON_HRT_DON', 'DON_NON_HRB_BEAT']: final_rename[col] = 'is_dcd'
        elif c_up in ['HEP_C_ANTI_DON', 'DON_ANTI_HCV']: final_rename[col] = 'hcv_status'
        elif c_up == 'KIL_DISPOSITION': final_rename[col] = 'KIL_DISPOSITION'
        elif c_up == 'KIR_DISPOSITION': final_rename[col] = 'KIR_DISPOSITION'

    df_donor = df_donor.rename(columns=final_rename)

    # 3. SPLIT INTO LEFT AND RIGHT KIDNEYS
    print("Transforming Donor-level data into Organ-level data...")
    if 'KIL_DISPOSITION' not in df_donor.columns or 'KIR_DISPOSITION' not in df_donor.columns:
        print("❌ CRITICAL: Could not identify Left/Right Kidney Disposition columns.")
        print("First 15 Columns found:", df_donor.columns.tolist()[:15])
        return
        
    # Split into independent rows
    df_left = df_donor.copy()
    df_left['disposition'] = df_left['KIL_DISPOSITION']
    
    df_right = df_donor.copy()
    df_right['disposition'] = df_right['KIR_DISPOSITION']
    
    df_kidneys = pd.concat([df_left, df_right], ignore_index=True)

    # 4. FILTER FOR RECOVERED ORGANS & CALCULATE TARGET
    print("Calculating true clinical utilization rate...")
    df_kidneys = df_kidneys.dropna(subset=['disposition'])
    df_kidneys['disposition'] = pd.to_numeric(df_kidneys['disposition'], errors='coerce')
    
    df_kidneys = df_kidneys[df_kidneys['disposition'].isin([5, 6])]
    df_kidneys['is_transplanted'] = (df_kidneys['disposition'] == 6).astype(int)
    
    success_rate = df_kidneys['is_transplanted'].mean()
    print(f"-> Target mapped successfully. True Historical Utilization Rate: {success_rate:.1%}")

    # 5. EXPORT DATASET
    print(f"\n✅ Success! Master dataset contains {df_kidneys.shape[0]} individual kidneys.")
    df_kidneys.to_csv(output_file, index=False)

if __name__ == "__main__":
    build_final_dataset(
        'DECEASED_DONOR_DATA.DAT', 'DECEASED_DONOR_DATA.htm', 
        'master_kidney_dataset_ALL_COLS.csv'
    )