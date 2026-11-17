import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file = pd.read_csv("./base.csv", header=0)
cols = list(file.columns)
print(cols)



#1 pega a coluna enviroment e mapeia ela para numeros inteiros ( awk -F "," '{print $11}' base.csv | sort | uniq -c )
aux = {'cold': 1, 'mild': 2, 'hot':3}
file['environment'] = file["environment"].map(aux)



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

    
    


