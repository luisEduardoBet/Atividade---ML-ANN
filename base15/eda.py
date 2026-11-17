import pandas as pd
import matplotlib.pyplot as plt

colunas = ['Feature_1', 'Feature_2', 'Label']
df = pd.read_csv('base-sintetica-15-labels.txt', sep='\t', names=colunas)



# Boxplots 
plt.figure(figsize=(10, 5))
df[['Feature_1', 'Feature_2']].boxplot()
plt.title('Boxplot das Features para Detecção de Outliers')
plt.ylabel('Valores')
plt.savefig("boxsplot.png") 

# Gráfico de Dispersão 
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Feature_1'], df['Feature_2'], c=df['Label'], cmap='tab20', alpha=0.7)
plt.title('Distribuição dos Dados por Classe (Features 1 e 2)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
#plt.colorbar(scatter, label='Classes (Labels)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("dispersao.png") 

print(df.info())

# Histogramas
df[['Feature_1', 'Feature_2']].hist(bins=30, figsize=(10, 5), edgecolor='black', alpha=0.7)
plt.suptitle('Distribuição dos Dados (Histogramas)')
plt.savefig("histograma.png")

#Correlação
correlacao = df[['Feature_1', 'Feature_2']].corr()
print(correlacao) 