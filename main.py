
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
os.makedirs("resultados", exist_ok=True)



wine = fetch_ucirepo(id=109)
X = wine.data.features
y = wine.data.targets.values.ravel()
execucoes = 20

acuracias = []

matriz_total = np.zeros((3, 3), dtype=int)


for i in range(execucoes): 
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=i,
        stratify=y
    )
    scaler = StandardScaler()

    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)


    modelo = MLPClassifier(
        hidden_layer_sizes=(10,), 
        activation="relu", 
        solver="adam",  
        max_iter=2000,  
        random_state=i  
    )

    
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, y_pred)


    acuracias.append(acuracia)

    matriz = confusion_matrix(y_teste, y_pred)
    matriz_total += matriz


   
    print(f"Execução {i+1}: {acuracia * 100:.2f}%")



print(f"=============================")
print(f"Melhor acurácia: {max(acuracias) * 100:.2f}%")


print(f"=============================")
print(f"Pior acurácia: {min(acuracias) * 100:.2f}%")


print(f"=============================")
print(f"Média de acurácia: {np.mean(acuracias) * 100:.2f}%")


print(f"=============================")
print(f"Desvio padrão: {np.std(acuracias) * 100:.2f}%")


plt.figure(figsize=(10, 5))


plt.plot(
    range(1, len(acuracias) + 1),
    np.array(acuracias) * 100,
    marker="o"
)



plt.title("Acurácia da MLP em 20 execuções")
plt.xlabel("Execução")
plt.ylabel("Acurácia (%)")
plt.grid(True)

plt.savefig("resultados/grafico_acuracia.png")
plt.close()


plt.figure(figsize=(8, 5))


plt.hist(
    np.array(acuracias) * 100,
    bins=6
)


plt.title("Distribuição das Acurácias da MLP")
plt.xlabel("Acurácia (%)")
plt.ylabel("Frequência")
plt.grid(True)

plt.savefig("resultados/histograma_acuracia.png")
plt.close()


plt.figure(figsize=(10, 5))


plt.bar(
    range(1, len(acuracias) + 1),
    np.array(acuracias) * 100
)


plt.title("Comparação das Acurácias por Execução")
plt.xlabel("Execução")
plt.ylabel("Acurácia (%)")
plt.grid(True)

plt.savefig("resultados/grafico_barras_acuracia.png")
plt.close()


plt.figure(figsize=(6, 5))


sns.heatmap(
    matriz_total,
    annot=True,
    fmt="d",
    cmap="Blues"
)


plt.title("Matriz de Confusão Total - 20 execuções")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")

plt.savefig("resultados/matriz_confusao.png")
plt.close()



print("\nArquivos gerados com sucesso:")
print("- resultados/grafico_acuracia.png")
print("- resultados/histograma_acuracia.png")
print("- resultados/grafico_barras_acuracia.png")
print("- resultados/matriz_confusao.png")