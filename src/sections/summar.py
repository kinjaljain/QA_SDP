import pickle
from collections import namedtuple

from summa import summarizer

import Levenshtein as lvstn

t = '''


Foma is a finite-state compiler, programming lan- guage, and regular expression/finite-state library designed for multi-purpose use with explicit sup- port for automata theoretic research, construct- ing lexical analyzers for programming languages, and building morphological/phonological analyz- ers, as well as spellchecking applications.
The compiler allows users to specify finite-state automata and transducers incrementally in a simi- lar fashion to AT&T’s fsm (Mohri et al., 1997) and Lextools (Sproat, 2003), the Xerox/PARC finite- state toolkit (Beesley and Karttunen, 2003) and the SFST toolkit (Schmid, 2005). One of Foma’s design goals has been compatibility with the Xe- rox/PARC toolkit. Another goal has been to al- low for the ability to work with n-tape automata and a formalism for expressing first-order logi- cal constraints over regular languages and n-tape- transductions.
Foma is licensed under the GNU general pub- lic license: in keeping with traditions of free soft- ware, the distribution that includes the source code comes with a user manual and a library of exam- ples.
The compiler and library are implemented in C and an API is available. The API is in many ways similar to the standard C library <regex.h>, and has similar calling conventions. However, all the low-level functions that operate directly on au- tomata/transducers are also available (some 50+ functions), including regular expression primitives and extended functions as well as automata deter- minization and minimization algorithms. These may be useful for someone wanting to build a sep- arate GUI or interface using just the existing low- level functions. The API also contains, mainly for spell-checking purposes, functionality for finding words that match most closely (but not exactly) a path in an automaton. This makes it straightfor- ward to build spell-checkers from morphological transducers by simply extracting the range of the transduction and matching words approximately.
Unicode (UTF8) is fully supported and is in fact the only encoding accepted by Foma. It has been successfully compiled on Linux, Mac OS X, and Win32 operating systems, and is likely to be portable to other systems without much effort.

'''


t2 = '''
The Foma project is multipurpose multi-mode finite-state compiler geared toward practical con- struction of large-scale finite-state machines such as may be needed in natural language process- ing as well as providing a framework for re- search in finite-state automata. Several wide- coverage morphological analyzers specified in the LEXC/xfst format have been compiled success- fully with Foma. Foma is free software and will remain under the GNU General Public License. As the source code is available, collaboration is encouraged.
A relative comparison of running a se- lection of regular expressions and scripts against other finite-state toolkits. The first and second en- tries are short regular expressions that exhibit ex- ponential behavior. The second results in a FSM with 22 states and 222 arcs. The others are scripts that can be run on both Xerox/PARC and Foma. The file lexicon.lex is a LEXC format English dic- tionary with 38418 entries. North Sami is a large lexicon (lexc file) for the North Sami language available from http://divvun.no.


'''








from tqdm import tqdm as tqdm

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')


import nltk
nltk.download('stopwords')


from rake_nltk import Rake
from nltk.corpus import stopwords
import re

stop = set(stopwords.words('english'))


root_path = "/Users/kai/PycharmProjects/QA_SDP2/src/dataloaders/"


with open("%sprocessed-data-2019-clean.pkl" % root_path, 'rb') as f:
  dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
  dataset2 = pickle.load(f)

# dataset =  dataset2 + dataset
dataset = dataset2


def get_id2_score(data):
    sents = data.ref.sentences.values()
    sents_merged = '\n'.join(sents)
    sent_scores = {x[0]: x[1] for x in summarizer.summarize(sents_merged, ratio=1, split=True, scores=True)}
    id2score = {}
    for id in data.ref.sentences:
        sentence = data.ref.sentences[id]
        if sentence in sent_scores:
            id2score[id] = sent_scores[sentence]
        else:
            closest_sent = sorted([x for x in sent_scores.keys()], key=lambda x: lvstn.distance(x, sentence), reverse=True)[0]
            id2score[id] = sent_scores[closest_sent]
    return id2score




paper = [data for data in dataset2 if data.ref.id == "E09-2008"]


v = get_id2_score(paper[0])



sents = [x for x in paper[0].ref.sentences.values()]

ref = '\n'.join(sents)
t = t2

t = ref
i = 0
sent_scores = {x[0]:x[1] for x in summarizer.summarize(t, ratio = 1, split=True, scores=True)}

data = paper[0]

score_id_map = {}


def get_closest(sentence, score_map):
    return sorted([x for x in score_map.keys()], key=lambda x: lvstn.distance(x, sentence))[0]



for id in data.ref.sentences:
    sentence = data.ref.sentences[id]
    if sentence in sent_scores:
        score_id_map[id] = sent_scores[sentence]
    else:
        score_id_map[id] = sent_scores[get_closest(sentence, sent_scores)]




print("ok")