import pandas as pd
from pathlib import Path
import json

p = Path(__file__).parent / 'CascadeModelSummary.csv'
print('Loading', p)
df = pd.read_csv(p)

# Fix count columns if incorrectly scaled
count_cols = ['TN','FP','FN','TP']
for col in count_cols:
    if col in df.columns:
        # If values are floats and small (e.g., <1000), scale up by 100
        mask = df[col].notna() & (df[col] < 1000)
        if mask.any():
            df.loc[mask, col] = (df.loc[mask, col] * 100).round().astype('Int64')

# Populate Macro F1 for Stage 2 rows from model artifacts when missing
for idx, row in df.iterrows():
    if row.get('Stage') and str(row['Stage']).strip().lower() == 'stage 2':
        if pd.isna(row.get('Macro F1')):
            model = str(row.get('Model')).strip()
            artifact = Path(__file__).parent.parent / 'model' / 'cascade' / model.lower() / 'stage2_metrics.json'
            if artifact.exists():
                try:
                    j = json.loads(artifact.read_text())
                    # try keys: validation.macro_f1 or validation['macro_f1'] or 'validation'->'macro_f1' or 'macro_f1'
                    val = None
                    if 'validation' in j and isinstance(j['validation'], dict):
                        val = j['validation'].get('macro_f1') or j['validation'].get('macro f1')
                    if val is None:
                        val = j.get('macro_f1') or j.get('macro f1')
                    if val is not None:
                        df.at[idx, 'Macro F1'] = float(val)
                        print(f"Filled Macro F1 for {model} Stage2 -> {val}")
                except Exception as e:
                    print('Failed to read', artifact, e)

# Ensure Macro F1 numeric
if 'Macro F1' in df.columns:
    df['Macro F1'] = pd.to_numeric(df['Macro F1'], errors='coerce')

# Save back
df.to_csv(p, index=False)
print('Saved cleaned CascadeModelSummary.csv')
print(df.dtypes)
print(df.to_string())
