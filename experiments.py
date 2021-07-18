# %%
import os

import gensim.models
import numpy as np
from algorithms.weat.lib import weat
from gensim.models import Word2Vec
from helper import load_json, remove_diac, remove_diac_dict
from algorithms.debiaswe.we import WordEmbedding, get_gender_spec
import algorithms.debiaswe.we as we
from sklearn.svm import LinearSVC
import json
from gensim.models import Word2Vec, KeyedVectors
from models import Word2VecLinearDebias
from helper import load_json, load_pkl, load_txt
import pickle as pkl


def load_m2f():
    with open('resources/male2female.dex', 'rb') as f:
        return pkl.load(f)


m2f = load_m2f()

word2vec = KeyedVectors.load_word2vec_format('word2vec_models/corpus_ndiacs.bin', binary=True)


# model=Word2VecLinearDebias()
def emb_in_dict(word2vec, d):
    words = word2vec.key_to_index
    present, missing = [], []
    for w in words:
        if w in d:
            present.append(w)
        else:
            missing.append(w)
    return present, missing


present, _ = emb_in_dict(word2vec, m2f)


def collect_present(d, present):
    return {w: d[w] for w in present}


valid_dict = collect_present(m2f, present)
