import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

# --- Configurações iniciais ---
st.set_page_config(page_title="Classificação Câncer de Mama", layout="wide")
st.title("Classificação de Câncer de Mama utilizando Aprendizagem de máquina")

# --- Carregar dados ---
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='diagnosis')
feature_names = data.feature_names
target_names = data.target_names

# --- Descrição ---
st.markdown("""
Aponta a classificação de tumores mamários com base no dataset **Breast Cancer Wisconsin (Diagnostic)** disponível para download no Kaggle.  
O modelo utiliza **aprendizado de máquina supervisionado** -que rotula os tumores como **maligno** e **benigno** utilizando dados já rotulados- especificamente um **Random Forest**.

- Número de amostras do dataset: **{}**
""".format(X.shape[0]))

# --- Treinar modelo ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))

# --- Gráfico de dispersão ---
st.header("Gráfico das Principais Variáveis")
df_plot = X[['mean radius', 'mean texture', 'mean perimeter', 'mean area']].copy()
df_plot['diagnosis'] = y.map({0: 'malignant', 1: 'benign'})
sns.set(style='ticks')
fig2 = sns.pairplot(df_plot, hue="diagnosis", diag_kind="hist", corner=True)
st.pyplot(fig2)

# --- Árvore individual ---
st.header("Visualização de uma das Árvores da Random Forest")

st.markdown(f"- Número de árvores usadas na Random Forest: **{len(clf.estimators_)}**")
st.markdown(f"- Amostras utilizadas no treino: **{len(X_train)}**")
st.markdown(f"- Amostras utilizadas no teste: **{len(X_test)}**")

st.markdown("Abaixo, exibimos **uma única árvore** da floresta aleatória treinada.")
tree_dot = export_graphviz(clf.estimators_[0],
                           feature_names=feature_names,
                           class_names=target_names,
                           filled=True, rounded=True,
                           special_characters=True)
st.graphviz_chart(tree_dot)

# --- Acurácia da árvore individual ---
tree = clf.estimators_[0]
tree_accuracy = accuracy_score(y_test, tree.predict(X_test))
st.info(f"Acurácia dessa árvore individual: **{tree_accuracy}**")

# --- Acurácia ---
st.success(f"Acurácia da Random Forest: **{accuracy}**")

# --- Matriz de confusão ---
st.header("Tabela de Erros e Acertos")
cm = confusion_matrix(y_test, clf.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
plt.xlabel("Valores Previstos")
plt.ylabel("Valores Reais")
st.pyplot(fig)

