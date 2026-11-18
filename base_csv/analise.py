import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file = pd.read_csv("./base.csv", header=0)
cols = list(file.columns)



#1 pega a coluna enviroment e mapeia ela para numeros inteiros ( awk -F "," '{print $11}' base.csv | sort | uniq -c )
aux = {'cold': 1, 'mild': 2, 'hot':3}
file['environment'] = file["environment"].map(aux)




#colocar apos remover os nulls e outliers
corr = file[cols].corr()
labels = cols
fig, ax = plt.subplots(figsize=(10, 8)) 
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
ax.set_yticks(range(len(labels)), labels=labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if abs(val) > 0.6 else "black")

fig.colorbar(im, ax=ax, label="Correlation")
fig.tight_layout()
fig.savefig("teste.png", dpi=200, bbox_inches="tight")
plt.close(fig)




for i in cols: 
    data = pd.to_numeric(file[i], errors="coerce")
    nulls = data.index[data.isnull()]
    not_nulls = data.dropna()  

    #Histograma
    plt.figure()
    plt.hist(not_nulls)
    plt.title(f"Distribuição Dados - {i}")
    plt.savefig(f"./histogramas/{i}.png")
    plt.close()

    # Boxplot
    plt.figure()
    plt.boxplot(not_nulls)
    plt.title(i)
    plt.savefig(f"./boxsplots/{i}.png")
    plt.close()
    
    


