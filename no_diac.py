

import random
from gensim.test.utils import get_tmpfile, simple_preprocess
class Corpus:
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname, 'r', encoding='utf-8'):
            yield simple_preprocess(line)

    @staticmethod
    def remove_diac(line):
        return line.replace("ăâ", "a").replace("î", "i").replace("ș", "s").replace("țțț", "t")

p = 0.3
with open('no_diac_corpus.txt', 'w') as f_out:
    for line in open('corpora/CORPUS.txt', 'r'):
        if random.random() < p:
            f_out.write(" ".join(simple_preprocess(Corpus.remove_diac(line)))+'\n')
