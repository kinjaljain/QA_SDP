import json
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import *
import numpy as np

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 16 if cuda else 1


class CiteDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


with open('prepared_data.json', 'r') as f:
    prepared_data = json.load(f)
train, valid = train_test_split(prepared_data, test_size=0.1)
train_dataset = CiteDataset(train)
valid_dataset = CiteDataset(valid)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=num_workers)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=256, num_workers=num_workers)


class CiteModel(nn.Module):
    def __init__(self):
        super(CiteModel, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.fc1 = nn.Linear(in_features=768 * 2, out_features=1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.1)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.1)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=512, out_features=2)

    def forward(self, ref_sentences, cite_sentences):
        ref_encoded = self.tokenizer.batch_encode_plus(ref_sentences, add_special_tokens=True)['input_ids']
        cite_encoded = self.tokenizer.batch_encode_plus(cite_sentences, add_special_tokens=True)['input_ids']

        ref_encoded = pad_to_max(ref_encoded)
        cite_encoded = pad_to_max(cite_encoded)
        ref_encoded, cite_encoded = torch.LongTensor(ref_encoded), torch.LongTensor(cite_encoded)
        x, ref_encoded = self.model(ref_encoded.to(device))
        print(x)
        print(ref_encoded)
        y, cite_encoded = self.model(cite_encoded.to(device))
        x = torch.cat([ref_encoded, cite_encoded], dim=1).to(device)
        x = self.act1(self.dropout1(self.bn1(self.fc1(x))))
        x = self.act2(self.dropout2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

def pad_to_max(sentences):
  max_len = max([len(x) for x in sentences])
  sentences = [np.pad(x, (0, max_len - len(x)), mode='constant', constant_values=0) for x in sentences]
  return sentences

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        num_correct = 0
        running_loss = 0.
        num_total = 0
        for batch_num, (ref, cite, label) in enumerate(train_loader):
            refs, cites, labels = ref.to(device), cite.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(refs, cites)
            _, predicted = torch.max(outputs.data, 1)
            num_total += label.size(0)
            num_correct += (predicted == label).sum().item()
            loss = criterion(outputs, label).detach()
            running_loss += loss.item()

        print('Train Accuracy: {}'.format(num_correct / num_total),
              'Average Loss: {}'.format(running_loss / len(train_loader)))

        model.eval()
        num_correct = 0
        running_loss = 0.
        num_total = 0
        with torch.no_grad():
            for batch_num, (ref, cite, label) in enumerate(valid_loader):
                refs, cites, labels = ref.to(device), cite.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(refs, cites)
                _, predicted = torch.max(outputs, 1)
                num_total += label.size(0)
                num_correct += (predicted == label).sum().item()
                loss = criterion(outputs, label).detach()
                running_loss += loss.item()

        print('Validation Accuracy: {}'.format(num_correct / num_total),
              'Average Loss: {}'.format(running_loss / len(valid_loader)))


model = CiteModel()
optimizer = optim.Adam(params=model.parameters())
num_epochs = 10
criterion = nn.CrossEntropyLoss()
train(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs)



