# nlp-laudos-medicos
## Análise de Similaridade em Laudos Médicos

Este projeto utiliza técnicas de NLP (Processamento de Linguagem Natural) e embeddings em português para analisar a similaridade entre expressões utilizadas em laudos laboratoriais e classificar automaticamente os resultados.

---

## Objetivo

Laudos médicos muitas vezes são escritos de forma livre e não padronizada, dificultando a análise automatizada. Este projeto:

- Converte frases livres em vetores semânticos.
- Mede a similaridade entre termos como “não detectado” e “não detectável”.
- Treina um modelo para classificar se o resultado é positivo ou negativo.

---

## Técnicas Utilizadas

- **Word Embeddings** (vetores de 100 dimensões em português)
- **Distância do Cosseno** para medir similaridade
- **Classificação com Random Forest**
- **Pipeline com Pandas, NumPy e Scikit-learn**

---

## Estrutura

```
nlp-laudos-medicos/
├── data/
│   └── base_resultados_1.xlsx
├── notebooks_antigos/
│   └── Trabalho_Final_NLP_Elisangela_Priscila.ipynb
├── main.py
├── requirements.txt
└── README.md
```

---

## Como Executar

```bash
# Clone o repositório
git clone https://github.com/larapri/nlp-laudos-medicos.git
cd nlp-laudos-medicos

# Instale os requisitos
pip install -r requirements.txt

# Execute o script
python nlp_laudos_similaridade.py

