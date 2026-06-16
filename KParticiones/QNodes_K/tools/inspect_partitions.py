import pandas as pd
from pathlib import Path
p = Path(__file__).resolve().parents[3] / 'QNodes' / 'results' / 'resultados_QNodes_k3_N10A.xlsx'
df = pd.read_excel(p)
print(df[['prueba','n_grupos']].to_dict('records'))
