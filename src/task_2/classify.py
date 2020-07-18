from collections import namedtuple

import pickle
import os

from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data.dataset import Dataset
from tqdm import tqdm_notebook as tqdm
import torch.nn.functional as F

from tqdm import tqdm as tqdm
from text_classify_utils import *

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')


from rake_nltk import Rake
from nltk.corpus import stopwords
import re

root_path = "/Users/kai/PycharmProjects/QA_SDP2/src/dataloaders/"

import json

accuracy = 0
total = 0
empty_citations = 0
empty_references = 0
import pickle


class CNN(nn.Module):
  def __init__(self, vocab_size, inp_size, out_size, embed_matrix, kernels):
    super(CNN, self).__init__()
    self.embed = nn.Embedding(vocab_size, inp_size)
    if embed_matrix is not None:
      self.embed.weight = nn.Parameter(torch.from_numpy(embed_matrix))
    self.convs = nn.ModuleList(
      [nn.Conv1d(1, number, (size, inp_size), padding=(size - 1, 0)) for (size, number) in kernels])
    self.dropout = nn.Dropout(0.2)
    self.out = nn.Linear(sum([x[1] for x in kernels]), out_size)

  def forward(self, x):
    x = self.embed(x)
    x = torch.stack([x], dim=1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x, 1)
    x = self.dropout(x)
    x = self.out(x)
    return x


with open("%sprocessed-data-2019-clean.pkl" % root_path, 'rb') as f:
    dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
    dataset2 = pickle.load(f)

dataset = dataset2

task2dataset = []
facet_map = {}
facet_count = {}
template_line = '''Citance Number: 1 | Reference Article:  ####REF####.txt | Citing Article:  ####CITE####.txt | Citation Marker Offset:  ####CITE_OFF#### | Citation Marker:  Green and Manning, 2010 | Citation Offset:  ['39'] | Citation Text:  <S sid ="39" ssid = "13">Joint segmentation and parsing was also investigated for Arabic (Green and Manning, 2010).</S> | Reference Offset:  ['1'] | Reference Text:  <S sid ="270" ssid = "63">6 Joint Segmentation and Parsing.</S> | Discourse Facet:  ####PREDICTION#### | Annotator:  Ankita Patel |'''

keys_2017 = []
for data in tqdm(dataset):
    citing_article = data.cite
    cite_id = citing_article.id
    facet = data.facet[0]
    ref_id = data.ref.id
    offsets = data.offsets
    citing_sentence_ids = offsets.cite
    section = citing_article.sections[citing_sentence_ids[0]]
    new_ids = [x for x in citing_sentence_ids]
    orig_facet = facet
    facet = facet.lower().replace("_", "").replace(" ", "").replace("results", "result")
    # Assuming they will fall in same section

    if facet not in facet_map:
        facet_map[facet] = len(facet_map.keys())

    keys = {"C98-1097", "D09-1023", "D10-1058", "N09-1001", "N09-1025", "P00-1025", "P07-1040", "W06-3909", "W09-0621",
            "W11-0815"}
    if ref_id in keys:
        is_2017 = True
    else:
        is_2017 = False
        for c in citing_sentence_ids:
            # If additional context is reqd
            to_add = 3
            extra = range(max(1, c - to_add), c)
            new_ids.extend(extra)
            extra = range(c + 1, min(len(citing_article.sentences), c + to_add + 1))
            new_ids.extend(extra)
    for c in set(new_ids):
        if is_2017:
            keys_2017.append((ref_id, cite_id, offsets.marker))
        complete_citing_sentence = citing_article.sentences[c]
        # TODO: Preprocessing
        facet_count[facet] = facet_count.get(facet, 0) + 1
        task2dataset.append([complete_citing_sentence.split(" "), facet_map[facet], section, is_2017, orig_facet])



task2dataset_train = [x for x in task2dataset if x[-2] == False]

task2dataset_eval = [x for x in task2dataset if x[-2] == True]

print(Counter([x[-1] for x in task2dataset_eval]))
print(Counter([x[1] for x in task2dataset_train]))
task2dataset = None
rev_facet = {facet_map[x]:x for x in facet_map}

print(facet_count)
print(facet_map)


from sklearn.metrics import classification_report, precision_recall_fscore_support
def evaluate(cnn, eval_set, should_output = False):
  cnn.eval()
  preds = []
  num_correct = 0
  num_valid_total = len(eval_set)
  eval_batch_iter = batch_iter(eval_set, batch_size=32, shuffle=False)
  all_preds = []
  all_reals = []
  with torch.no_grad():
    for batch in eval_batch_iter:
        sentences = [e[0] for e in batch]
        labels = [e[1] for e in batch]
        word_ids = to_input_tensor(
            sentences, word_to_id,
            pad_index=word_to_id['<pad>'], unk_index=word_to_id['<unk>']
        ).to(device)
        labels = torch.from_numpy(np.array(labels)).to(device)
        outputs = cnn.forward(word_ids)
        loss = criterion(outputs, labels)
        _, y_pred = torch.max(outputs, 1)
        all_preds.extend(y_pred.detach().tolist())
        all_reals.extend(labels.detach().tolist())
        num_correct += (y_pred == labels).sum().item()
  print("Eval Accuracy : ", num_correct/num_valid_total)
  #print(classification_report(all_reals, all_preds))
  F1metrics = precision_recall_fscore_support(all_reals, all_preds, average='macro')
  # print('Precision: ', F1metrics[0])
  # print('Recall: ', F1metrics[1])
  print('MACRO F1score:', F1metrics[2])
  F1metrics = precision_recall_fscore_support(all_reals, all_preds, average='micro')
  # print('Precision: ', F1metrics[0])
  # print('Recall: ', F1metrics[1])
  print('MICRO F1score:', F1metrics[2])

  if should_output:
      with open('task2_preds.txt', 'w') as f:

          for idx, pred in enumerate(all_preds):
              ref_id, cite_id, marker = keys_2017[idx]
              marker = str([str(x) for x in (marker)])
              facet_text = rev_facet[pred]
              line = template_line.replace('####PREDICTION####', facet_text).replace("####REF####", ref_id).replace(
                  "####CITE####", cite_id).replace("####CITE_OFF####", marker)
              f.write(line + "\n")
      print("Written")

"""
Macro F1 => F1 for a, F1 for b, F1 for c , 0, 0=> a+b+c/3


"""




word_to_id = build_vocab(task2dataset_train, lambda x: x[0])
max_vocab_size = len(word_to_id)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

cnn = CNN(max_vocab_size, 300, len(facet_map), None, [(3,30), (4,30), (5,30)])
cnn.to(device)
optimizer = torch.optim.Adam(cnn.parameters(), weight_decay=1e-3)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([5.07, 1, 6.94, 21.50, 7.22]))
# TODO: Class weights
batch_size = 32
batch_p = 10
num_total = len(task2dataset_train)

for epoch_id in range(500):
    num_correct = 0
    train_batch_iter = batch_iter(task2dataset_train, batch_size=batch_size, shuffle=True)
    cnn.train()
    for batch_id, batch in enumerate(train_batch_iter):
        optimizer.zero_grad()
        sentences = [e[0] for e in batch]
        labels = [e[1] for e in batch]
        word_ids = to_input_tensor(
            sentences, word_to_id,
            pad_index=word_to_id['<pad>'], unk_index=word_to_id['<unk>']
        ).to(device)
        labels = torch.from_numpy(np.array(labels)).to(device)
        outputs = cnn.forward(word_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, y_pred = torch.max(outputs, 1)
        num_correct += (y_pred == labels).sum().item()
        if batch_id % batch_p == batch_p - 1:
            print("Batch number %d acc %.3f"%(batch_id, num_correct/(batch_id*batch_size + batch_size)))

    evaluate(cnn, task2dataset_eval, True)
    evaluate(cnn, task2dataset_train)


print("ok")

"""
Set 1 :
C00-2123
C02-1025
C04-1089
C08-1098
C10-1045
Set 2:
C90-2039
C94-2154
C98-1097
D09-1023
D10-1058
Set 3:
D10-1083
E03-1020
E09-2008
H05-1115
H89-2014
Set 4:
I05-5011
J00-3003
J96-3004
J98-2005
N01-1011

Kai - Set 1 and 3
Kinjal - Set 1 and 2
Anjana - Set 2 and 4
Riki - Set 3 and 4
"""