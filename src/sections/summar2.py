import logging
import pickle
from collections import namedtuple

import pytextrank
import spacy
import sys


Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')


root_path = "/Users/kai/PycharmProjects/QA_SDP2/src/dataloaders/"


with open("%sprocessed-data-2019-clean.pkl" % root_path, 'rb') as f:
  dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
  dataset2 = pickle.load(f)

# dataset =  dataset2 + dataset
dataset = dataset2

paper = [data for data in dataset2 if data.ref.id == "E09-2008"]

sents = [x for x in paper[0].ref.sentences.values()]
ref = '\n'.join(sents)


text = ref

nlp = spacy.load("en_core_web_sm")
tr = pytextrank.TextRank(logger=None)

nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
doc = nlp(text)

for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=99999):
    print(sent)

