import pandas as pd
import numpy as np
import os
import pymprog as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, display_html
import time

DATA_PATH = "../dados/"
FILE = "inst_25.txt"


def get_files(data_path):
    data = {}
    for file in sorted(os.listdir(data_path)):
        if not file.endswith(".txt"):
            continue

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
    return data


data = get_files(DATA_PATH)


# Functions
def euclidian_distance(i, j, nodes):
    return np.sqrt((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2)


def render_nodes(df):
    nodes = []
    for index, row in df.iterrows():
        nodes.append((row["X"], row["Y"], row["Tempo de Serviço"], row["Deadline"]))

    nodes.append((nodes[0][0], nodes[0][1], nodes[0][2], nodes[0][3]))
    return nodes


def beauty_print(text, char="="):
    text_size = len(text)
    print(char * text_size)
    print(text)
    print(char * text_size)


# Functions


fig, ax = plt.subplots(1, len(data), figsize=(20, 5))
limit = 100

key = "inst_25.txt"

for index, row in data[key].iterrows():
    ax[0].plot(row["X"], row["Y"], "o")
    # ax[0].Circle((row["X"], row["Y"]), row["Tempo de Serviço"], color="r", radius=0.03 * 100)
    ax[0].set_title(key)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y", rotation=0)
    ax[0].grid(False)
    ax[0].axis("equal")
    ax[0].set_aspect("equal")
    ax[0].set_xlim(0, limit)
    ax[0].set_ylim(0, limit)
    ax[0].set_xticks(np.arange(0, limit + 1, 10))
    ax[0].set_yticks(np.arange(0, limit + 1, 10))

plt.show()


#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################
#############################################################################################################################################################

import pymprog as pm

file = "inst_25.txt"

# Iniciando o modelo
start_time = time.time()
pm.begin("Caixeiro viajante com deadline")


# parâmetros
nodes = render_nodes(data[file])
M = (
    data[file]["Deadline"].sum() + 10
)  # M suficientemente grande para que a solução seja ótima
service_times = data[file]["Tempo de Serviço"].to_list()
deadlines = data[file]["Deadline"].to_list()


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

# Eliminação de subrotas
for i in range(nodes_count - 1):
    for j in range(1, nodes_count):
        phi[j] >= phi[i] + (service_times[i] + euclidian_distance(i, j, nodes)) * x[
            i, j
        ] - M * (1 - x[i, j])

for j in range(1, nodes_count - 1):
    w[j] >= phi[j] - deadlines[j]

# bagulhinho magico pra caixeiro viajante
pm.solver(int, gmi_cuts=1)
pm.solver(int, tm_lim=3600000 * 2) 

pm.solve()
end_time = time.time()
pm.save(
    mps="../dados/results/inst_25/inst_25.mps",
    sol="../dados/results/inst_25/inst_25.sol",
    clp="../dados/results/inst_25/inst_25.clp",
    glp="../dados/results/inst_25/inst_25.glp",
    sen="../dados/results/inst_25/inst_25.sen",
    ipt="../dados/results/inst_25/inst_25.ipt",
    mip="../dados/results/inst_25/inst_25.mip",
)


# printando solução
time.sleep(1)
print("=" * 100)
print(f"\n\tTempo de execução: {end_time - start_time:.2f}s")
print(f"\tSolução ótima (Acumulo de atraso): {pm.vobj():.2f}")

i = 0
j = 1
cont = 0
print(f"\n\tPercurso dos nós")

while cont < nodes_count - 1:
    while x[i, j].primal < 0.9:
        j += 1
    print("\t{:2d}   -->  {:2d} | Atraso: {:.2f}".format(i, j, w[j].primal))
    i = j
    j = 0
    cont += 1

print("\n", "=" * 100)

# plotando a solução
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect="equal")
# ax1.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
ax1.grid(False)
limit = 100

for i, j in cartesian_product:
    if i != j:
        if x[i, j].primal >= 0.9:
            ax1.add_patch(
                patches.ConnectionPatch(
                    xyA=(nodes[i][0], nodes[i][1]),
                    xyB=(nodes[j][0], nodes[j][1]),
                    coordsA="data",
                    coordsB="data",
                    color="r",
                )
            )

for i in range(0, nodes_count - 1):
    ax1.add_patch(
        patches.Circle((nodes[i][0], nodes[i][1]), radius=0.03 * limit, color="#F4A460")
    )
    plt.text(
        nodes[i][0] - len(str(i)) * 0.02 * limit / 2,
        nodes[i][1] - 0.02 * limit / 2,
        str(i),
        {"color": "black", "fontsize": 10},
    )

ax1.add_patch(
    patches.Circle((nodes[0][0], nodes[0][1]), radius=0.04 * limit, color="r")
)
plt.text(
    nodes[0][0] - 0.02 * limit / 2,
    nodes[0][1] - 0.02 * limit / 2,
    "D",
    {"color": "black", "fontsize": 10},
)
plt.ylim(-1, limit + 1)
plt.xlim(-1, limit + 1)

fig1.show()

plt.savefig("../dados/results/inst_25/inst_25_result.png")
plt.show(block=True)
plt.interactive(False)
