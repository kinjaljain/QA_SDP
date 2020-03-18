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
        x = self.data[item]
        return x[0], x[1], x[2]


with open('prepared_data.json', 'r') as f:
    prepared_data = json.load(f)

new_prep = []
new_prep.append(prepared_data[0])
new_prep.append(prepared_data[7])
# prepared_data = new_prep
# train, valid = prepared_data, prepared_data

train, valid = train_test_split(prepared_data, test_size=0.1)
train_dataset = CiteDataset(train)
valid_dataset = CiteDataset(valid)
batch_size = 1
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 8 if cuda else 1
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)


class CiteModel(nn.Module):
    def __init__(self):
        super(CiteModel, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.fc1 = nn.Linear(in_features=768, out_features=1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.1)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        # self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.1)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=512, out_features=2)

    def forward(self, ref_sentences, cite_sentences):
        ref_encoded = self.tokenizer.batch_encode_plus([x.lower() for x in ref_sentences], add_special_tokens=True)['input_ids']
        cite_encoded = self.tokenizer.batch_encode_plus([x.lower() for x in cite_sentences], add_special_tokens=True)['input_ids']
        # print(ref_encoded)
        # print(cite_encoded)
        # sdfc
        # ref_encoded = pad_to_max(ref_encoded)
        # cite_encoded = pad_to_max(cite_encoded)
        # print(ref_encoded)
        # print(cite_encoded)
        # sdfc
        ref_encoded, cite_encoded = torch.LongTensor(ref_encoded), torch.LongTensor(cite_encoded)
        inputs = torch.cat([ref_encoded, cite_encoded], dim = 1).to(device)
        token_tensor0 = torch.zeros_like(ref_encoded).to(device)
        token_tensor1 = torch.ones_like(cite_encoded).to(device)
        token_tensor = torch.cat([token_tensor0, token_tensor1], dim = 1).to(device)
        print(token_tensor.shape)
        print(inputs.shape)
        # print(cite_encoded.shape)
        # x, ref_encoded = self.model(ref_encoded.to(device))
        # y, cite_encoded = self.model(cite_encoded.to(device))
        _, x = self.model(inputs, token_type_ids = token_tensor)
        # x = torch.cat([ref_encoded, cite_encoded], dim=1).to(device)
        x = self.act1(self.dropout1(self.fc1(x)))
        x = self.act2(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        return x

def pad_to_max(sentences):
  max_len = max([len(x) for x in sentences])
  sentences = [np.pad(x, (0, max_len - len(x)), mode='constant', constant_values=3) for x in sentences]
  return sentences

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}:".format(epoch))
        model.train()
        num_correct = 0
        running_loss = 0.
        num_total = 0
        for batch_num, d in enumerate(train_loader):
            refs, cites, labels = d[0], d[1], d[2].to(device)
            # print(len(d[0]))
            # print(len(labels))
            if batch_num % 1 == 0:
                optimizer.zero_grad()
            outputs = model(refs, cites)
            _, predicted = torch.max(outputs, 1)
            num_total += labels.size(0)
            num_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            if batch_num % 1 == 0:
                optimizer.step()
            if batch_num % 100 == 0:
              print("acc : ", (num_correct)/num_total)

        print('Train Accuracy: {}'.format(num_correct / num_total),
              'Average Loss: {}'.format(running_loss / len(train_loader)))

        model.eval()
        num_correct = 0
        running_loss = 0.
        num_total = 0
        with torch.no_grad():
            for batch_num, d in enumerate(valid_loader):
                refs, cites, labels = d[0], d[1], d[2].to(device)
                outputs = model(refs, cites)
                _, predicted = torch.max(outputs, 1)
                num_total += labels.size(0)
                num_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        print('Validation Accuracy: {}'.format(num_correct / num_total),
              'Average Loss: {}'.format(running_loss / len(valid_loader)))


device = torch.device("cuda" if cuda else "cpu")
model = CiteModel()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr = 1e-4)
num_epochs = 5000
criterion = nn.CrossEntropyLoss()
train(model, train_dataloader, valid_dataloader, optimizer, criterion, num_epochs)

torch.save(model.state_dict(), 'model.npy')
torch.save(optimizer.state_dict(), 'optimizer.npy')
