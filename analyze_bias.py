# %%
import os

import gensim.models
import numpy as np
from algorithms.weat.lib import weat
from gensim.models import Word2Vec
from helper import load_json,remove_diac,remove_diac_dict
from algorithms.debiaswe.we import WordEmbedding, get_gender_spec
import algorithms.debiaswe.we as we
from sklearn.svm import LinearSVC
import json

MODEL_DIR = 'word2vec_models/'
model_paths = [MODEL_DIR + f for f in os.listdir('word2vec_models/')]
weat_paths = os.listdir('algorithms/weat_resources/')


def get_word_vectors(words, model):
    """
    Returns word vectors represent words
    :param words: iterable of words
    :return: (len(words), dim) shaped numpy ndarrary which is word vectors
    """
    words = [w for w in words if w in model.words]
    return [model.v(w) for w in words]

class WeatPrompt():
    def __init__(self, dirname):
        self.file = open(dirname, 'r', encoding='utf-8')
        self.prompt = self.unpack_prompt()

    def unpack_prompt(self):
        prompt = {}
        for line, set_label in zip(self.file, ['X', 'Y', 'A', 'B']):
            contents = list(map(lambda x: x.strip(), line.split(':')[1].split(',')))
            prompt[set_label] = contents
        return prompt

    def __str__(self):
        return json.dumps(self.prompt)


class Weat:
    def __init__(self, paths):
        self.paths = paths

    @staticmethod
    def eval_weat(paths, model):
        score = {}
        for path in paths:
            prmpt = WeatPrompt('algorithms/weat_resources/' + path).prompt
            if 'băiat' not in model.words:
                new_prmpt = remove_diac_dict(prmpt)

            X, Y, A, B = list(map(lambda x: get_word_vectors(x,model), (prmpt[x] for x in "XYAB")))
            bias_weat = weat.weat_score(X, Y, A, B)
            score[path] = bias_weat
        return score


class Word2VecModel:
    def __init__(self, model_path, num_training, wpaths):
        self.old_model = WordEmbedding(model_path)
        self.model = WordEmbedding(model_path)
        print(f'OLD EVALUATION \n{Weat.eval_weat(wpaths,self.old_model)}')
        self.load_debias(num_training)
        self.debias()
        print(f'NEW EVALUATION\n{Weat.eval_weat(wpaths, self.model)}')

    def load_debias(self, num_training):
        defs_path = "algorithms/debiaswe/data/definitional_pairs.json"
        equalize_path = 'algorithms/debiaswe/data/equalize_pairs.json'
        gender_seed_path = 'algorithms/debiaswe/data/gender_specific_seed.json'
        defs, equalize, gender_seed = list(
            map(load_json, [defs_path, equalize_path, gender_seed_path]))

        debias_dict = {'defs': defs, 'equalize': equalize, 'gender_spec': gender_seed}
        if 'fată' not in self.model.words:
            debias_dict = remove_diac_dict(debias_dict)

        gender_spec = get_gender_spec(debias_dict['gender_spec'], num_training, self.model)
        debias_dict['gender_spec'] = gender_spec
        self.debias_dict = debias_dict

    def debias(self):
        gender_specific_words, definitional, equalize = [self.debias_dict[name] for name in
                                                         ['gender_spec', 'defs', 'equalize']]
        gender_direction = we.doPCA(definitional, self.model).components_[0]
        specific_set = set(gender_specific_words)
        for i, w in enumerate(self.model.words):
            if w not in specific_set:
                self.model.vecs[i] = we.drop(self.model.vecs[i], gender_direction)
        self.model.normalize()
        candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower())]}
        for (a, b) in candidates:
            if a in self.model.index and b in self.model.index:
                y = we.drop((self.model.v(a) + self.model.v(b)) / 2, gender_direction)
                z = np.sqrt(1 - np.linalg.norm(y) ** 2)
                if (self.model.v(a) - self.model.v(b)).dot(gender_direction) < 0:
                    z = -z
                self.model.vecs[self.model.index[a]] = z * gender_direction + y
                self.model.vecs[self.model.index[b]] = -z * gender_direction + y
        self.model.normalize()

#%%
for path in model_paths:
    if path.endswith('.bin'):
        print(path)
        model = Word2VecModel(path, 10000, wpaths=weat_paths)



#%%
model=Word2VecModel(model_paths[1],10000,wpaths=weat_paths)
model=model.old_model
#%%
print(model.more_words_like_these(['barba   t']))
diff=model.diff('profesor','invatator') #Invatator invatatoare profeor
model.best_analogies_dist_thresh(diff,max_words=40000,topn=100)
#%%
import gensim
model=gensim.models.KeyedVectors.load_word2vec_format(model_paths[1],binary=True)
# %%
from helper import word_like, load_word2lemma, remove_diac

word2lemma = load_word2lemma()
word2lemma_nd = {}
for key, value in word2lemma.items():
    tkey = remove_diac(key)
    word2lemma_nd[tkey] = value
word = "băiat"
print(word_like(word, word2lemma))
# %%
from helper import preproc
import random
from gensim.utils import simple_preprocess

with open('corpora/oscar_merged.txt', 'r', encoding='utf-8') as f, open('oscar_trimmed11.txt', 'w',
                                                                        encoding='utf-8') as g:
    for line in f:
        if random.random() < 0.3:
            line = " ".join(simple_preprocess(line))
            g.write(line + '\n')
#%%
def lemmatize(word2lemma,line):
    new_line=""
    for word in line:
        if word in word2lemma:
            new_line+=word2lemma[word]+" "
        else:
            new_line+=word +" "
    return new_line
#%%
from gensim.utils import simple_preprocess
with open('corpora/oscar_merged.txt','r',encoding='utf-8') as f,open('corpora/oscar_merged_clean','w',encoding='utf-8') as g:
    for line in f:
        line=" ".join(simple_preprocess(line))
        g.write(line+'\n')
