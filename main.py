#=====================
#Imports

# Essas bibliotecas são usadas para manipulação de dados e gráficos.
#=====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


#=====================
# download do dataset Wine diretamente do repositório do UCI
#=====================
from ucimlrepo import fetch_ucirepo


#=====================
# Divide os dados em treino e teste.
#=====================
from sklearn.model_selection import train_test_split


#=====================
# normalização dos dados
#=====================
from sklearn.preprocessing import StandardScaler


#=====================
# Cria a rede neural MLP (Multi-layer Perceptron).
#=====================
from sklearn.neural_network import MLPClassifier


#=====================
# Calcura a acurácia e a matriz de confusão.
#=====================
from sklearn.metrics import accuracy_score, confusion_matrix


#=====================
# Cria a pasta resultados, caso ela não exista
#=====================
os.makedirs("resultados", exist_ok=True)


#=====================
# Carregando o dataset Wine
# Aqui o Python baixa o dataset Wine.   
#=====================
wine = fetch_ucirepo(id=109)

#=====================
# X recebe as entradas, ou seja, os dados químicos do vinho.
# álcool, magnésio, cor, prolina...
#=====================
X = wine.data.features


#=====================
# y recebe a resposta correta, ou seja, a classe do vinho (1, 2 ou 3).
# ex: Classe 1, 2 ou 3.
#=====================
y = wine.data.targets.values.ravel()


#=====================
# Configs iniciais
# Define que o modelo será treinado 20 vezes.
#=====================
execucoes = 20


#=====================
# Cria uma lista vazia para guardar a acurácia de cada execução.
#=====================
acuracias = []


#=====================
# Cria uma matriz 3x3 zerada para acumular a matriz de confusão de todas as execuções.
#Ela será usada para somar as matrizes de confusão das 20 execuções.
#=====================
matriz_total = np.zeros((3, 3), dtype=int)

#=====================
# Laço de repetição para treinar e avaliar o modelo 20 vezes.
# Esse for faz o código repetir 20 vezes.
# Como execucoes = 20, ele vai rodar da execução 1 até a execução 20.
#=====================
for i in range(execucoes): 

    #=====================
    # Separando treino e teste
    # Aqui os dados são divididos em:
    # 70% para treino
    # 30% para teste

    #test_size=0.30 significa que 30% dos dados serão usados para teste e o restante (70%) para treino.
    #random_state=i faz cada exexução ter uma divião diferente dos dados.
    # stratify=y mantém a proporção das classes nos conjuntos de treino e teste.
    # Isso evita, por exemplo, que uma classe fique quase toda no treino e pouca no teste.
    #=====================
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=i,
        stratify=y
    )


    #=====================
    # Normalização dos dados
    # Cria o normalizador.
    #======================
    scaler = StandardScaler()


    #=====================
    # Calcula a média e o desvio padrão e normaliza os dados de treino.
    #=====================
    X_treino = scaler.fit_transform(X_treino)

    #=====================
    # Normaliza os dados de teste usando a mesma regra do treino.
    # Isso é importante porque a MLP trabalha melhor com os dados em escalar parecidas.
    #=====================
    X_teste = scaler.transform(X_teste)


    #=====================
    # Criando a rede neural MLP

    # Aqui a rede neural é criada.
    #======================
    modelo = MLPClassifier(
        hidden_layer_sizes=(10,),  # A rede tem uma camada oculta com 10 neurônios
        activation="relu",  # Define a função de ativação ReLU, a ReLU é uma função que ajuda a rede a aprender padrões não lineares
        solver="adam",  # Define o algoritmo de treinamento como Adam. Ele ajusta os pesos da rede automaticamente durante o treinamento.
        max_iter=2000,  # Define o máximo de 2000 épocas de treinamento.
        random_state=i  # Faz cada execução ter uma inicialização diferente, mas controlada.
    )

    #=====================
    # Treinando o modelo.

    # Aqui a MLP aprende com os dados de treino.
    # Ela recebe:
    # X_treino = características dos vinhos
    # y_treino = classes corretas dos vinhos
    # Durante o treinamento, a rede ajusta seus pesos para tentar acertar as classes.
    #=====================
    modelo.fit(X_treino, y_treino)


    #=====================
    # Testando o modelo.

    # Depois de treinar, o modelo tenta prever as classes dos vinhos do conjunto de teste.
    # y_pred guarda as respostas previstas pela rede.
    #=====================
    y_pred = modelo.predict(X_teste)


    #=====================
    # Calculando a acurácia

    # Compara:
    # y_teste = resposta corretas
    # y_pred = resposta prevista pela rede
    # E calcula o percentual de acertos.
    #=====================
    acuracia = accuracy_score(y_teste, y_pred)


    #=====================
    # Guarda a acurácia da execução atual na lista.
    #=====================
    acuracias.append(acuracia)


    #=====================
    # Matriz de confusão

    # Cria a matriz de confusão para a execução atual.
    # Ela mostra quantos exemplos foram classificados corretamente e quantos foram confundidos entre as classes.
    #=====================
    matriz = confusion_matrix(y_teste, y_pred)


    #=====================
    # Soma a matriz atual com a matriz total das execuções anteriores.
    #=====================
    matriz_total += matriz


    #=====================
    # Mostrando o resultado de cada execução.
    # Mostra no terminal a acurácia de cada execução.   

    # Exemplo:
    # Execução 1: 94.44%
    # Execução 2: 98.15%
    #=====================
    print(f"Execução {i+1}: {acuracia * 100:.2f}%")


#=====================
# Resultados finais

# Mostra a melhor acurácia obtida na média das 20 execuções.
#=====================
print(f"=============================")
print(f"Melhor acurácia: {max(acuracias) * 100:.2f}%")


#=====================
# Mostra a pior acurácia.
#=====================
print(f"=============================")
print(f"Pior acurácia: {min(acuracias) * 100:.2f}%")


#=====================
# Mostra a acurácia das 20 execuções.
#=====================
print(f"=============================")
print(f"Média de acurácia: {np.mean(acuracias) * 100:.2f}%")


#=====================
# Mostra o quanto os resultados variaram.
#=====================
print(f"=============================")
print(f"Desvio padrão: {np.std(acuracias) * 100:.2f}%")


#=====================
# Gráfico das acurácias
# Cria a area do gráfico
#=====================
plt.figure(figsize=(10, 5))


#=====================
# Cria um gráfico de linha mostrando a acurácia de cada execução.
#=====================
plt.plot(
    range(1, len(acuracias) + 1),
    np.array(acuracias) * 100,
    marker="o"
)


#=====================
# Define título, nomes dos eixos, grade, salva e mostra o gráfico.
#=====================
plt.title("Acurácia da MLP em 20 execuções")
plt.xlabel("Execução")
plt.ylabel("Acurácia (%)")
plt.grid(True)

plt.savefig("resultados/grafico_acuracia.png")
plt.close()


#=====================
# Histograma das acurácias
# Mostra a distribuição das acurácias obtidas nas 20 execuções.
# Permite visualizar a frequência com que determinados percentuais de acerto ocorreram.
#=====================
plt.figure(figsize=(8, 5))


#=====================
# Cria um histograma mostrando a distribuição das acurácias.
#=====================
plt.hist(
    np.array(acuracias) * 100,
    bins=6
)


#=====================
# Define título, nomes dos eixos, grade, salva e mostra o gráfico.
#=====================
plt.title("Distribuição das Acurácias da MLP")
plt.xlabel("Acurácia (%)")
plt.ylabel("Frequência")
plt.grid(True)

plt.savefig("resultados/histograma_acuracia.png")
plt.close()


#=====================
# Gráfico de barras das acurácias
# Mostra a acurácia obtida em cada execução através de barras.
# Facilita a comparação entre os resultados.
#=====================
plt.figure(figsize=(10, 5))


#=====================
# Cria um gráfico de barras mostrando a acurácia de cada execução.
#=====================
plt.bar(
    range(1, len(acuracias) + 1),
    np.array(acuracias) * 100
)


#=====================
# Define título, nomes dos eixos, grade, salva e mostra o gráfico.
#=====================
plt.title("Comparação das Acurácias por Execução")
plt.xlabel("Execução")
plt.ylabel("Acurácia (%)")
plt.grid(True)

plt.savefig("resultados/grafico_barras_acuracia.png")
plt.close()


#=====================
# Gráfico da matriz de confusão
# Cria um mapa de calor da matriz de confusão total.
#=====================
plt.figure(figsize=(6, 5))


#=====================
# Cria um mapa de calor da matriz de confusão total.
#=====================
sns.heatmap(
    matriz_total,
    annot=True,
    fmt="d",
    cmap="Blues"
)


#=====================
# Coloca título, nomes dos eixos, salva e mostra o gráfico.
#=====================
plt.title("Matriz de Confusão Total - 20 execuções")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")

plt.savefig("resultados/matriz_confusao.png")
plt.close()


#=====================
# Mostra quais arquivos foram gerados
#=====================
print("\nArquivos gerados com sucesso:")
print("- resultados/grafico_acuracia.png")
print("- resultados/histograma_acuracia.png")
print("- resultados/grafico_barras_acuracia.png")
print("- resultados/matriz_confusao.png")