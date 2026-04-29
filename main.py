import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# dados
df = pd.read_csv("transacoes.csv")

# normalização
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['valor_transacao']])

# modelo
model = IsolationForest(contamination=0.1, random_state=42)
df['anomalia'] = model.fit_predict(df_scaled)

# resultado
print(df)

# gráfico
plt.scatter(df.index, df['valor_transacao'], c=df['anomalia'])
plt.title("Detecção de Anomalias")
plt.show()