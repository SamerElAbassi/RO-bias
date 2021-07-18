import itertools
import nltk
import numpy as np
import spacy
import pickle as pkl
import re
import nltk
from gensim.models import Word2Vec, KeyedVectors
import json


class BatchIter():
    def __init__(self, obj, size):
        self.obj = obj
        self.size = size

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError
        return self.obj[idx * self.size:(idx + 1) * self.size]

    def __len__(self):
        return len(self.obj) // self.size


def store_pkl(obj, fname):
    with open(fname, 'wb') as f:
        pkl.dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return pkl.load(fname)


def load_txt(fname):
    with open(fname, 'r') as f:
        return f.readlines()


def word_like(w_like, voc, topn=1):
    distances = [nltk.edit_distance(w_like, w) for w in perm_diac(w_like) if w in voc]
    print(distances)


def remove_fsuffx():
    pass


def perm_diac(word):
    diacs = "ăâîșț"
    diac_map = {'â': 'a',
                'ă': 'a',
                'î': 'i',
                'ț': 't',
                'ș': 's',
                }

    mask = np.ones(len(word))
    poz = [idx for idx, letter in enumerate(word) if letter in diacs]

    perms = []
    for num in range(1, len(poz) + 1):
        perms.extend(list(itertools.combinations(poz, num)))

    words = []
    for perm in perms:
        t = np.copy(mask)
        t[np.array(perm)] = 0

        tword = "".join([letter if t[idx] == 1 else diac_map[word[idx]] for idx, letter in enumerate(word)])
        words.append(tword)

    return words


def load_word2vec(fname):
    if fname.endswith('.bin'):
        return KeyedVectors.load_word2vec_format(fname, binary=True)
    if fname.endswith('.model'):
        return Word2Vec.load(fname).wv


def load_word2lemma():
    with open('resources/word2lemma.pkl', 'rb') as f:
        word2lemma = pkl.load(f)
        f.close()
    return word2lemma


def complete_lemma_voc(batch_text, word2lemma):
    nlp = spacy.load('ro_core_news_lg')
    for batch in batch_text:
        buffered = []
        for w in batch:
            if w not in word2lemma:
                buffered.append(w)
        new_words = nlp(" ".join(buffered), disable=['ner', 'parser'])
        for w in new_words:
            word2lemma[w.text] = w.lemma_

    return word2lemma


def remove_diac(line):
    return line.replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t")


def unpack_text_batches(batches):
    return "\n".join(batches)


def load_json(fname):
    with open(fname, "r", encoding='utf-8') as f:
        return json.load(f)


def preproc(text, stops):
    """

    :param text: String
    :return: Makes text lower and removes any nonalphanumeric
    """
    text = text.lower()
    tokenized_text = " ".join([w for w in " ".join(text.split('-')).split() if w.isalpha()])
    return tokenized_text


def remove_diac_list(l):
    t = [list(map(remove_diac, w)) if type(w) == list else remove_diac(w) for w in l]
    return t


def remove_diac_dict(d):
    for key, value in d.items():
        d[key] = remove_diac_list(value)
    return d


def translate_text_with_model(text, target="ro", model="nmt"):
    translate_client = translate.Client.from_service_account_json(CREDENTIALS)
    result = translate_client.translate(text, target_language=target, model=model)
    return result['translatedText']


def find_word_in_corpus(iter_text, word):
    """

    :param iter_text: Iterable of paragraph-like texts (reviews, comments, statuses etc)
    :param word: Word to look for
    :return: Returns list of sentences containing the word in a paragraph.
    """
    all_relevant_sentences = []

    for text in iter_text:
        relevant_sentences = []
        for sentence in text.split('.'):
            try:
                if sentence.split().index(word):
                    relevant_sentences.append(sentence)
            except ValueError:
                pass

        if relevant_sentences:
            all_relevant_sentences.append(relevant_sentences)

    return all_relevant_sentences


def rand_select_secv(iter_text, prob):
    lines = []
    for line in iter_text:
        if random.random() < prob:
            lines.append(line)
    return lines
