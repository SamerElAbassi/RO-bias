# %%
from gensim.models import Word2Vec
import logging
import multiprocessing
from gensim.test.utils import get_tmpfile, simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from helper import unpack_text_batches
import multiprocessing
import re
from helper import BatchIter
import nltk
from nltk.corpus import stopwords
from helper import preproc
from gensim.test.utils import get_tmpfile, simple_preprocess
import pickle as pkl
import random
import spacy
from tqdm import trange

# nlp = spacy.load('ro_core_news_lg')

stops = stopwords.words('romanian')
from dataset import IterCorpus
CORPUS_ROOT = "corpora/oscar_lemmatized"
CORPUS = IterCorpus(CORPUS_ROOT)
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence

CORPUS = LineSentence('oscar_merged_cleaned1.txt')
#%%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = Word2Vec(corpus_file='corpora/oscar_merged_clean', vector_size=300, window=5, min_count=500,
                 workers=8)

# selected_texts = random.sample(text, text_len // 2)
# selected_texts=load_pkl('selected_texts_short.pkl')
# del text
# %%
model.wv.save_word2vec_format('Oscar_Merged_Cleaned.bin', binary=True)
