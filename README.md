# Fundamentos de Inteligência Artificial

## Sobre o Projeto

Este projeto foi desenvolvido como atividade de Trabalho de Desenvolvimento de Equipe (TDE) da disciplina de Fundamentos de Inteligência Artificial.

O objetivo é aplicar conceitos de Redes Neurais Artificiais utilizando uma Rede Neural Perceptron Multicamadas (MLP - Multi Layer Perceptron) para classificação de padrões a partir do conjunto de dados Wine Dataset.

O projeto realiza o treinamento da rede neural, avalia seu desempenho em múltiplas execuções e apresenta métricas estatísticas e gráficos para análise dos resultados obtidos.

---

## Objetivos

* Implementar uma Rede Neural Artificial do tipo MLP.
* Aplicar técnicas de treinamento supervisionado.
* Avaliar o desempenho do modelo através da acurácia.
* Realizar múltiplas execuções para análise estatística.
* Gerar gráficos para visualização dos resultados.

---

## Dataset Utilizado

O conjunto de dados utilizado foi o Wine Dataset.

Características:

* 178 amostras
* 13 atributos numéricos
* 3 classes de vinho

---

## Tecnologias Utilizadas

* Python 3.x
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* UCI ML Repository

---

## Estrutura do Projeto

```text
TDE_FundamentosIA/
│
├── main.py
├── README.md
├── resultados/
│   ├── grafico_acuracia.png
│   ├── histograma_acuracia.png
│   ├── grafico_barras_acuracia.png
│   └── matriz_confusao.png
└── Relatorio_TDE.pdf
```

---

## Instalação

Clone o repositório:

```bash
git clone https://github.com/Tiago-MdrS/TDE_FundamentosIA.git
```

Entre na pasta:

```bash
cd TDE_FundamentosIA
```

Instale as dependências:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
```

---

## Execução

Execute o projeto com:

```bash
python main.py
```

---

## Resultados Obtidos

Durante a execução são realizadas 20 rodadas de treinamento da rede neural.

Ao final são apresentados:

* Melhor acurácia
* Pior acurácia
* Média das acurácias
* Desvio padrão

Exemplo:

```text
Melhor acurácia: 100.00%
Pior acurácia: 88.89%
Média de acurácia: 97.22%
Desvio padrão: 1.99%
```

---

## Gráficos Gerados

### Evolução das Acurácias

Mostra o desempenho da rede neural em cada execução.

### Histograma das Acurácias

Apresenta a distribuição dos resultados obtidos ao longo das 20 execuções.

### Comparação das Acurácias

Permite comparar visualmente a acurácia alcançada em cada treinamento.

### Matriz de Confusão

Permite visualizar os acertos e erros de classificação da rede neural.

---

## Conceitos Aplicados

* Inteligência Artificial
* Redes Neurais Artificiais
* Perceptron Multicamadas (MLP)
* Aprendizado Supervisionado
* Classificação Multiclasse
* Padronização de Dados (StandardScaler)
* Métricas de Avaliação
* Matriz de Confusão

---

## Autores

Tiago Madeira Silva
Lindomar Marcos Pinheiro Neto

Disciplina: Fundamentos de Inteligência Artificial
Curso: Análise e Desenvolvimento de Sistemas
