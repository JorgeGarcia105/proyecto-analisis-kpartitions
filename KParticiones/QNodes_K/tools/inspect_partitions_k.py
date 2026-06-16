import sys
from pathlib import Path
import pandas as pd
k = int(sys.argv[1]) if len(sys.argv)>1 else 3
p = Path(__file__).resolve().parents[3] / 'QNodes' / 'results' / f'resultados_QNodes_k{k}_N10A.xlsx'
df = pd.read_excel(p)
print('file=',p)
print(df[['prueba','n_grupos','particion']].head(5).to_dict('records'))
