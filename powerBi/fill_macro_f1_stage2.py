import pandas as pd
from pathlib import Path
import json

p = Path(__file__).parent / 'CascadeModelSummary.csv'
print('Loading', p)
df = pd.read_csv(p)

model_map = {
    'random forest': 'randomforest',
    'support vector machine': 'svm',
    'xgboost': 'xgboost'
}

for idx, row in df.iterrows():
    if str(row.get('Stage')).strip().lower() == 'stage 2':
        if pd.isna(row.get('Macro F1')):
            model = str(row.get('Model')).strip().lower()
            folder = model_map.get(model, model.replace(' ','').lower())
            artifact = Path(__file__).parent.parent / 'model' / 'cascade' / folder / 'stage2_metrics.json'
            if artifact.exists():
                try:
                    j = json.loads(artifact.read_text())
                    val = None
                    if 'validation' in j and isinstance(j['validation'], dict):
                        val = j['validation'].get('macro_f1') or j['validation'].get('macro f1')
                    if val is None:
                        val = j.get('macro_f1') or j.get('macro f1')
                    if val is not None:
                        df.at[idx, 'Macro F1'] = float(val)
                        print(f"Filled Macro F1 for {row.get('Model')} Stage2 -> {val}")
                except Exception as e:
                    print('Failed to read', artifact, e)

# Save
p.write_text(df.to_csv(index=False))
print('Saved updated CascadeModelSummary.csv')
print(df[['Model','Stage','Macro F1']])
