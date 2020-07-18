from collections import namedtuple

from transformers import pipeline

nlp = pipeline('question-answering')

import pickle



Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')


root_path = "/Users/kai/PycharmProjects/QA_SDP2/src/dataloaders/"


with open("%sprocessed-data-2019-clean.pkl" % root_path, 'rb') as f:
  dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
  dataset2 = pickle.load(f)


data = [x for x in dataset2 if x.ref.id == "E09-2008"][0]
print(data.cite.id)
sents = data.ref.sentences.values()
sents_merged = '. \n'.join([x for x in sents])

effect = nlp({
        'question' : 'Is it open source?'.lower(),
        'context' : sents_merged.lower()
    })
print(effect)
