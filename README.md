#  TDE - Fundamentos de Inteligência Artificial

##  Sobre o Projeto

Este projeto foi desenvolvido como atividade de Trabalho de Desenvolvimento de Equipe (TDE) da disciplina de Fundamentos de Inteligência Artificial.

O objetivo é aplicar conceitos de Redes Neurais Artificiais utilizando uma Rede Neural Perceptron Multicamadas (MLP - Multi Layer Perceptron) para classificação de padrões a partir do conjunto de dados Wine Dataset.

O projeto realiza o treinamento da rede neural, avalia seu desempenho em múltiplas execuções e apresenta métricas estatísticas e gráficos para análise dos resultados obtidos.

---

##  Objetivos

- Implementar uma Rede Neural Artificial do tipo MLP.
- Aplicar técnicas de treinamento supervisionado.
- Avaliar o desempenho do modelo através da acurácia.
- Realizar múltiplas execuções para análise estatística.
- Gerar gráficos para visualização dos resultados.

---

##  Dataset Utilizado

O conjunto de dados utilizado foi o Wine Dataset, disponível na biblioteca Scikit-Learn.

Características:

- 178 amostras
- 13 atributos numéricos
- 3 classes de vinho

---

##  Tecnologias Utilizadas

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

##  Estrutura do Projeto

text
TDE_FundamentosIA/
│

├── main.py

├── wine.data

├── grafico_acuracia.png

├── grafico_classes.png

├── grafico_matriz_confusao.png

├── README.md

└── Relatorio_TDE.pdf

---

## Instalação

Clone o repositório:

git clone https://github.com/Tiago-MdrS/TDE_FundamentosIA.git

Entre na pasta:

cd TDE_FundamentosIA

## Instale as dependências:

pip install numpy pandas matplotlib scikit-learn


## Execute o projeto com:

python main.py

## Durante a execução são realizadas 20 rodadas de treinamento da rede neural.

Ao final são apresentados:

Melhor acurácia
Pior acurácia
Média das acurácias
Desvio padrão

exemplo:

Melhor acurácia: 100.00%
Pior acurácia: 88.89%
Média de acurácia: 95.74%
Desvio padrão: 2.93%
 Gráficos Gerados
Evolução das Acurácias

