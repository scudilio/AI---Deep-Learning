"""Gera data/word2vec_reviews_pt.npz: vetores word2vec treinados nos reviews do B2W-Reviews01.

Como foi feito (reproduz o arquivo distribuído com a disciplina, a menos da semente aleatória):
1. Corpus: B2W-Reviews01 (github.com/americanas-tech/b2w-reviews01, CC BY-NC-SA 4.0), coluna review_text,
   todos os 132.373 reviews (não só os 30 mil do subconjunto da aula).
2. Tokenização: minúsculas e apenas sequências de letras (com acentos): regex [a-záàâãéêíóôõúç]+.
   Números e pontuação são descartados. Nada de stemming, stopwords ou dicionário.
3. Modelo: gensim Word2Vec, CBOW (sg=0), 100 dimensões, janela 5, min_count=5 (a palavra precisa
   aparecer em pelo menos 5 reviews), 5 épocas. Com min_count=5 o vocabulário fica em 13.607 palavras.
4. Saída: um .npz com dois arrays: 'vocab' (13.607 palavras, ordenadas da mais frequente para a menos)
   e 'vetores' (13.607 x 100, float32).

Treino: cerca de 10 a 15 segundos em CPU. A semente e o número de threads mudam um pouco os vizinhos
(na nossa verificação, 22 dos 30 vizinhos de 5 palavras-teste coincidiram entre dois treinos);
a estrutura (sinônimos, erros de digitação, aritmética) é a mesma.

Uso:
    pip install gensim pandas
    python gerar_word2vec.py caminho/para/B2W-Reviews01.csv
"""
import re
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

CSV = sys.argv[1] if len(sys.argv) > 1 else "B2W-Reviews01.csv"
SAIDA = "word2vec_reviews_pt.npz"

df = pd.read_csv(CSV, low_memory=False)
textos = df["review_text"].dropna().astype(str)
frases = [re.findall(r"[a-záàâãéêíóôõúç]+", t.lower()) for t in textos]
print(len(frases), "reviews tokenizados")

modelo = Word2Vec(frases, vector_size=100, window=5, min_count=5, sg=0, epochs=5, workers=4, seed=42)
vocab = modelo.wv.index_to_key                     # ordenado por frequência
vetores = modelo.wv[vocab].astype(np.float32)
np.savez_compressed(SAIDA, vocab=np.array(vocab), vetores=vetores)
print("Salvo:", SAIDA, "|", len(vocab), "palavras x", vetores.shape[1], "dimensões")
print("Vizinhos de 'entrega':", [w for w, _ in modelo.wv.most_similar("entrega", topn=6)])
