import pandas as pd
p='CascadeModelSummary.csv'
df=pd.read_csv(p)
print('DF dtypes:')
print(df.dtypes)
print('\nFull table:')
print(df.to_string(index=False))
print('\nStage2 rows:')
st2=df[df['Stage'].str.strip().str.lower()=='stage 2']
print(st2[['Model','Stage','Macro F1','Validation Accuracy','Validation F1 Score']].to_string(index=False))
print('\nAverages for Macro F1 by Model (Stage 2):')
print(st2.groupby('Model')['Macro F1'].mean())
print('\nAll Macro F1 values:')
print(df['Macro F1'].tolist())
