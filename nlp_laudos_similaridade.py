# 📁 Projeto: Análise de Similaridade em Laudos Médicos

"""
Este notebook implementa um pipeline de NLP para análise de similaridade semântica entre termos clínicos extraídos de laudos laboratoriais.

Objetivos:
1. Carregar os dados de laudos médicos não padronizados.
2. Aplicar embeddings em português para vetorização semântica.
3. Calcular similaridade de termos utilizando distância do cosseno.
4. Treinar um modelo simples de classificação para prever categorias clínicas.

Tecnologias:
- Embeddings Word2Vec em português
- Pandas, NumPy, Scikit-learn
- Métricas: precisão, revocação, acurácia
"""

# 🔧 Importação de bibliotecas
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 📥 Carregando os dados (simulado para este exemplo)
data = {
    'Divulgado': ["não detectado", "não detectável", "reagente", "detectado", "influenza", "h1n1"],
    'Resultado': ["negativo", "negativo", "positivo", "positivo", "positivo", "positivo"]
}
df = pd.DataFrame(data)

# 🧹 Pré-processamento simples
def preprocess(text):
    return text.lower().strip()

df['Divulgado_proc'] = df['Divulgado'].apply(preprocess)

# 📌 Carregar embeddings pré-treinados (exemplo simulado)
# Substitua por: model = KeyedVectors.load_word2vec_format("path/to/embedding.vec")
class DummyEmbedding:
    def __init__(self):
        self.vocab = {"não": np.random.rand(100), "detectado": np.random.rand(100), "detectável": np.random.rand(100),
                      "reagente": np.random.rand(100), "influenza": np.random.rand(100), "h1n1": np.random.rand(100)}

    def __getitem__(self, key):
        return self.vocab.get(key, np.zeros(100))

embedding_model = DummyEmbedding()

# 🔎 Obter vetor médio por frase
def embed_text(text):
    tokens = text.split()
    vectors = [embedding_model[token] for token in tokens if token in embedding_model.vocab]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

df['embedding'] = df['Divulgado_proc'].apply(embed_text)

# 🔁 Similaridade entre termos
similaridade = cosine_similarity([df['embedding'][0]], [df['embedding'][1]])[0][0]
print(f"Similaridade entre '{df['Divulgado'][0]}' e '{df['Divulgado'][1]}': {similaridade:.2f}")

# 🤖 Classificação com embeddings
X = np.vstack(df['embedding'].values)
y = df['Resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📊 Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))
