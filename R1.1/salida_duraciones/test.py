import pandas as pd

d = pd.read_csv('R1.1/salida_duraciones/duraciones_por_archivo.csv', sep=';', encoding='utf-8-sig')
print(d['gender'].value_counts(normalize=True).round(4)*100)
print(d[d['categoria']=='OK']['gender'].value_counts(normalize=True).round(4)*100)