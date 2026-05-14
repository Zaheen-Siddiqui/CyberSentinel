import pandas as pd
from pathlib import Path

p = Path(__file__).parent / 'CascadeModelSummary.csv'
print('Loading', p)
df = pd.read_csv(p, dtype=str)

# Strip whitespace from all string cells
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# Columns to coerce to numeric (if present)
num_cols = [
    'Validation Accuracy', 'Validation F1 Score', 'Macro F1',
    'False Positive Rate (Normal)', 'Attack Recall',
    'TN', 'FP', 'FN', 'TP'
]

for col in num_cols:
    if col in df.columns:
        # Replace empty strings with NaN
        df[col] = df[col].replace({'': None, '': None})
        # Remove percentage signs and convert
        df[col] = df[col].str.replace('%','', regex=False)
        df[col] = df[col].str.replace(',','', regex=False)
        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # If values look like percentages > 1 (e.g., 99.89), convert to fraction
        mask = df[col] > 1
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col] / 100.0

# Save back
out = p
df.to_csv(out, index=False)
print('Cleaned and saved', out)
print(df.dtypes)
print(df.head().to_string())
