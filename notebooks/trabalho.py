# %%
# %pip install pandas numpy pymprog matplotlib 

# %%
import pandas as pd 
import numpy as np 
import os 
import pymprog as p
import matplotlib.pyplot as plt
from IPython.display import display, display_html

# %%
data_path = "../dados/"


# %% [markdown]
# # Leitura dos arquivos
# 

# %%
data = {}

for file in sorted(os.listdir(data_path)):
    if not file.endswith(".txt"):
        continue

    print(file)
    with open(os.path.join(data_path, file), "r") as f:
        row = []
        for idx, line in enumerate(f):
            if idx == 0:  # ignora a quantidae de localidades
                continue

            row.append([int(x) for x in line.split()])
        data[file] = pd.DataFrame(
            row, columns=["X", "Y", "Tempo de Serviço", "Deadline"]
        )
        data[file].fillna(0, inplace=True)


# %% [markdown]
# # Exibindo os dados

# %%
fig, ax = plt.subplots(1, len(data), figsize=(20, 5))

for idx, key in enumerate(data.keys()):
    display(data[key])

    for index, row in data[key].iterrows():
        ax[idx].plot(row["X"], row["Y"], "o")
        ax[idx].set_title(key)
        ax[idx].set_xlabel("X")
        ax[idx].set_ylabel("Y", rotation=0)
        ax[idx].grid(True)
        ax[idx].axis("equal")
        ax[idx].set_aspect("equal")
        ax[idx].set_xlim(0, 100)
        ax[idx].set_ylim(0, 100)
        ax[idx].set_xticks(np.arange(0, 101, 10))
        ax[idx].set_yticks(np.arange(0, 101, 10))


plt.show()


# %% [markdown]
# # Elaborando modelo

# %% [markdown]
# # Funções

# %%
def euclidian_distance(i, j, nodes):
    return np.sqrt((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2)


def render_nodes(df):
    nodes = []
    for index, row in df.iterrows():
        nodes.append((row["X"], row["Y"], row["Tempo de Serviço"], row["Deadline"]))

    nodes.append((nodes[0][0], nodes[0][1], nodes[0][2], nodes[0][3]))
    return nodes


# %% [markdown]
# # Modelo

# %%
nodes, deadlines

# %%
import pymprog as pm

# Iniciando o modelo
pm.begin("Caixeiro viajante com deadline")
display(data["inst_10.txt"])


# parâmetros
nodes = render_nodes(data["inst_10.txt"])
M = (
    data["inst_10.txt"]["Deadline"].sum() + 10
)  # M suficientemente grande para que a solução seja ótima
service_times = data["inst_10.txt"]["Tempo de Serviço"].to_list()
deadlines = data["inst_10.txt"]["Deadline"].to_list()


# variáveis de decisão
nodes_count = len(nodes)
cartesian_product = pm.iprod(range(nodes_count), range(nodes_count))
x = pm.var("x", cartesian_product, bool)
phi = pm.var("phi", nodes_count)  # acumulo de tempo / inicio de serviço no nó
w = pm.var("w", nodes_count)  # atraso de cada rota


# função objetivo
pm.minimize(sum(w[j] for j in range(nodes_count - 1)))

# restrições
for i in range(nodes_count - 1):
    sum(x[i, j] for j in range(1, nodes_count) if i != j) == 1

for j in range(1, nodes_count):
    sum(x[i, j] for i in range(nodes_count - 1) if i != j) == 1

# elminando sub-rotas
# for i in range(nodes_count - 1):
#     for j in range(1, nodes_count):
#         phi[j] >= phi[i] + nodes_count * x[i, j] - nodes_count + 1

for i in range(nodes_count - 1):
    for j in range(1, nodes_count):
        phi[j] >= phi[i] + (service_times[i] + euclidian_distance(i, j, nodes)) * x[
            i, j
        ] - M * (1 - x[i, j])

for j in range(1, nodes_count-1):
    w[j] >= phi[j] - deadlines[j]

# bagulhinho magico pra caixeiro viajante
pm.solver(int, gmi_cuts=1)

pm.solve()

# printando solução
print("\nSolução ótima:", pm.vobj())
for i in range(nodes_count - 1):
    for j in range(1, nodes_count):
        if x[i, j].primal > 0.5:
            print(i, j)

for j in range(1, nodes_count-1):
    print(phi[j].primal, deadlines[j], phi[j].primal-deadlines[j])


i = 0
j = 1
cont = 0
while cont < nodes_count-1:
  while x[i,j].primal < 0.9:
    j += 1
  print("{:2d}   -->  {:2d}".format(i,j))
  i = j
  j = 0
  cont += 1




