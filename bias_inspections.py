# %%
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import logging
import multiprocessing
from gensim.test.utils import get_tmpfile, simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing
import re
import numpy as np
import scipy

model = Word2Vec.load('word2vec_300d_10eps.model').wv
MAX_WORDS = 10000
vectors = model.vectors[:MAX_WORDS]
# Normalize vectors -> Make them unit vectors
norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
id2word = model.index_to_key
dots = norm_vectors.dot(norm_vectors.T)  # Get words that fit together
selected_words = scipy.sparse.csr_matrix(dots * (dots >= 1 - 1 / 2))
selected_words,scores = selected_words.nonzero(),selected_words.data
#%%
np.set_printoptions(2)
for row,col,score in zip(*selected_words,scores):
    print(f'{id2word[row]} -> {id2word[col]} with score {score:.3f}')

# %%
for row in dots:
    sum = np.sum(row ** 2)
    print(sum)
    if max > 0.1:
        print(max)
# %%
diacs = 'ăâîșț'  # For simple use
MW = 'bărbat'
FW = 'femeie'
print(model.most_similar([MW, 'ieftin'], [FW], topn=30))


# %%


def related_word_embs(model, word):
    data = []
    for token, _ in model.most_similar(word, topn=10):
        data.append((token, model[token]))
    return data


relevant_words = ['barbat', 'femeie', 'angajat', 'scriitoare']
related_words = [pair for word in relevant_words for pair in related_word_embs(model, word)]
# %%
tokens = []
embs = []
for token, emb in related_words:
    tokens.append(token)
    embs.append(emb)
# %%
tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=2500, random_state=23)

new_vals = tsne_model.fit_transform(embs)
import matplotlib.pyplot as plt

for idx, val in enumerate(new_vals):
    plt.scatter(val[0], val[1])
    plt.annotate(tokens[idx], xy=val)
plt.show()
# %%
import spacy


nlp=spacy.load('ro_core_news_sm')
lines=open('no_diac_corpus.txt','r',encoding='utf-8').readlines()
#%%
for line in lines:
    sent=nlp(line)
    print(sent.text)
    for w in sent:
        print(w.lemma_,end=" ")
    print()
#%%
model = Word2Vec.load('word2vec_300d_10eps.model').wv
