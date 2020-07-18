import pickle
from collections import namedtuple

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from text_classify_utils import build_vocab, batch_iter, to_input_tensor

Datum = namedtuple('Datum', 'ref cite offsets author is_test facet year')
Offsets = namedtuple('Offsets', 'marker cite ref')
Article = namedtuple('Article', 'id xml sentences sections')

root_path = "/Users/kai/PycharmProjects/QA_SDP2/src/dataloaders/"

class CNN(nn.Module):
  def __init__(self, vocab_size, inp_size, out_size, embed_matrix, kernels):
    super(CNN, self).__init__()
    self.embed = nn.Embedding(vocab_size, inp_size)
    if embed_matrix is not None:
      self.embed.weight = nn.Parameter(torch.from_numpy(embed_matrix), requires_grad=True)
    self.convs = nn.ModuleList(
      [nn.Conv1d(1, number, (size, inp_size), padding=(size - 1, 0)) for (size, number) in kernels])
    self.dropout = nn.Dropout(0.25)
    self.fc1 = nn.Linear(sum([x[1] for x in kernels]), 128)
    self.out1 = nn.Linear(128, 2)
    self.out2 = nn.Linear(128, 2)
    self.out3 = nn.Linear(128, 2)

  def forward(self, x):
    x = self.embed(x)
    x = torch.stack([x], dim=1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x, 1)
    x = self.dropout(x)
    x = self.fc1(x)

    return self.out1(x), self.out2(x)


def get_group(ref_section, groups):
    for idx, group in enumerate(groups):
        for ele in group:
            if ele in ref_section.lower():
                return idx
    return -1


groups = [("abstract", "intro", "concl", "paper", "summ"), ("",)]

with open("%sprocessed-data-2019-clean.pkl" % root_path, 'rb') as f:
    dataset = pickle.load(f)

with open("%sprocessed-data-2018-clean.pkl" % root_path, 'rb') as f:
    dataset2 = pickle.load(f)
total = 0
rep = 0
g = 0

dataset2 = dataset + dataset2

num_outs = len(groups)

ys = []
X = []
totals = [0 for _ in range(num_outs)]
for data in dataset2:
    ref_article = data.ref
    cite_article = data.cite
    ref_offset = data.offsets.ref
    ref_sections = [ref_article.sections[x] for x in ref_offset]
    ref_sections_orig = ref_sections
    ref_groups = [get_group(ref_section, groups) for ref_section in ref_sections]
    ref_groups = set(ref_groups)
    # print(ref_sections)
    ref_sections = ref_groups
    if len((ref_sections)) > 1:
        print(ref_sections_orig)
        print(ref_sections)
        rep += 1
    if len((ref_sections)) == 1:
        g += 1
    total += 1
    y = [0 for _ in range(num_outs)]
    for val in ref_sections:
        y[val] = 1
        totals[val] += 1
    ys.append(y)
    complete_citing_sentence = ''.join([cite_article.sentences[c] for c in data.offsets.cite])
    X.append(complete_citing_sentence.split())

Y = np.array(ys)
print("totals", totals)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
X_train, X_test, Y_train, Y_test = X[:-753],X[-753:], Y[:-753], Y[-753:]


word_to_id = build_vocab(X_train, lambda x: x)
max_vocab_size = len(word_to_id)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

cnn = CNN(max_vocab_size, 300, num_outs, None, [(3,30), (4,30), (5,30)])
cnn.to(device)
optimizer = torch.optim.Adam(cnn.parameters(), weight_decay=1e-3)


c1 = nn.CrossEntropyLoss(weight=torch.tensor([1, 2]).float().to(device))
c2 = nn.CrossEntropyLoss(weight=torch.tensor([1, 8]).float().to(device))
c3 = nn.CrossEntropyLoss(weight=torch.tensor([1, 2]).float().to(device))
batch_size = 32
batch_p = 10
num_total = len(X_train)


print(num_total , " examples = ", num_total/batch_size , " batches")


from sklearn.metrics import classification_report, precision_recall_fscore_support

def evaluate(cnn, eval_set, should_output = False):
  cnn.eval()
  num_valid_total = len(eval_set)
  eval_batch_iter = batch_iter(eval_set, batch_size=32, shuffle=False)
  all_preds = []
  all_reals = []
  nc1 = 0
  nc2 = 0
  nc3 = 0
  num_correct = 0
  with torch.no_grad():
    for batch in eval_batch_iter:
        optimizer.zero_grad()
        sentences = [e[0] for e in batch]
        labels = torch.tensor(np.array([e[1] for e in batch]))
        word_ids = to_input_tensor(
            sentences, word_to_id,
            pad_index=word_to_id['<pad>'], unk_index=word_to_id['<unk>']
        ).to(device)
        labels = labels.to(device)
        l1, l2= labels[:, 0], labels[:, 1]

        o1, o2 = cnn.forward(word_ids)

        _, y_pred1 = torch.max(o1, 1)
        _, y_pred2 = torch.max(o2, 1)
        # _, y_pred3 = torch.max(o3, 1)
        nc1 += (y_pred1 == l1).sum().item()
        nc2 += (y_pred2 == l2).sum().item()
        # nc3 += (y_pred3 == l3).sum().item()
        all_preds.extend(zip(y_pred1.detach().tolist(), y_pred2.detach().tolist(), y_pred2.detach().tolist()))
        all_reals.extend(zip(l1.detach().tolist(), l2.detach().tolist(), l2.detach().tolist()))
  print("Valid = ", should_output)
  print("Eval Accuracy : ", np.array([nc1, nc2, nc3])/num_valid_total)
  all_preds = np.array(all_preds).T
  all_reals = np.array(all_reals).T
  print("-" * 100)
  for preds, reals in zip(all_preds, all_reals):

        #print(classification_report(all_reals, all_preds))
        F1metrics = precision_recall_fscore_support(reals, preds, average='binary')
        # print('Precision: ', F1metrics[0])
        # print('Recall: ', F1metrics[1])
        print('MACRO F1score:', F1metrics[2])
        print("*"*100)
  print("-"*100)


for epoch_id in range(500):
    nc1 = 0
    nc2 = 0
    nc3 = 0
    running_loss = 0
    num_correct = 0
    train_batch_iter = batch_iter([(x,y) for x,y in zip(X_train, Y_train)], batch_size=batch_size, shuffle=True)
    cnn.train()
    for batch_id, batch in enumerate(train_batch_iter):
        optimizer.zero_grad()
        sentences = [e[0] for e in batch]
        labels = torch.tensor(np.array([e[1] for e in batch]))
        word_ids = to_input_tensor(
            sentences, word_to_id,
            pad_index=word_to_id['<pad>'], unk_index=word_to_id['<unk>']
        ).to(device)
        labels = labels.to(device)
        l1,l2 = labels[:,0], labels[:,1]

        o1, o2 = cnn.forward(word_ids)
        loss1 = c1(o1, l1)
        loss2 = c2(o2, l2)
        # loss3 = c3(o3, l3)

        _, y_pred1 = torch.max(o1, 1)
        _, y_pred2 = torch.max(o2, 1)
        # _, y_pred3 = torch.max(o3, 1)
        nc1 += (y_pred1 == l1).sum().item()
        nc2 += (y_pred2 == l2).sum().item()
        # nc3 += (y_pred3 == l3).sum().item()

        loss = loss1 + loss2 #+ loss3
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

        if batch_id % batch_p == batch_p - 1:
            print("Batch number", (batch_id), "acc", (np.array([nc1, nc2, nc3])/(batch_id*batch_size + batch_size)), running_loss/(batch_id*batch_size+batch_size))

    evaluate(cnn, [(x,y) for x,y in zip(X_test, Y_test)], True)
    evaluate(cnn, [(x,y) for x,y in zip(X_train, Y_train)])




print(rep, total, g)


