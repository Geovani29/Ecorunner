# generate_reports.py
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# 1) ε vs episodio
df = pd.read_csv(os.path.join(DATA_DIR, "metrics.csv"))
plt.figure()
plt.plot(df["episode"], df["epsilon"])
plt.xlabel("Episodio")
plt.ylabel("Epsilon")
plt.title("Decaimiento de ε por episodio")
plt.show()

# 2) Recompensa media en bloques de 1000
df_r = pd.read_csv(os.path.join(DATA_DIR, "rewards.csv"))
block_size = 1000
means = [df_r["total_reward"][i:i+block_size].mean() 
        for i in range(0, len(df_r), block_size)]
plt.figure()
plt.plot(range(1, len(means)+1), means)
plt.xlabel("Bloque de 1000 episodios")
plt.ylabel("Recompensa media")
plt.title("Recompensa media por bloque de 1000 episodios")
plt.show()

# 3) Histograma de estados visitados
df_s = pd.read_csv(os.path.join(DATA_DIR, "state_counts.csv"))
# convertir el string de tupla a tupla
df_s["state"] = df_s["state"].apply(eval)
counts = df_s["count"].values
plt.figure()
plt.bar(range(len(counts)), counts)
plt.xlabel("Estado (codificado)")
plt.ylabel("Número de visitas")
plt.title("Distribución de estados visitados")
plt.show()
