import pandas as pd
from pathlib import Path

p = Path(__file__).parent / 'cyberssentinel_powerbi_combined.csv'
if not p.exists():
    p = Path(__file__).parent.parent / 'powerBi' / 'cyberssentinel_powerbi_combined.csv'
    # fallback

if not p.exists():
    raise FileNotFoundError(f"CSV not found: {p}")

df = pd.read_csv(p)

print('CSV:', p)
print('Rows, cols:', df.shape)
print('\nPer-model Attack Recall (TP / (TP + FN))')
for m, g in df.groupby('model_used'):
    tp = int(((g['actual_binary'] == 1) & (g['prediction_numeric'] == 1)).sum())
    fn = int(((g['actual_binary'] == 1) & (g['prediction_numeric'] == 0)).sum())
    total_attacks = int((g['actual_binary'] == 1).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"{m}: TP={tp} FN={fn} TotalAttacks={total_attacks} Recall={recall:.4f}")

# overall
all_tp = int(((df['actual_binary'] == 1) & (df['prediction_numeric'] == 1)).sum())
all_fn = int(((df['actual_binary'] == 1) & (df['prediction_numeric'] == 0)).sum())
all_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
print('\nOverall Attack Recall:', round(all_recall, 4))
