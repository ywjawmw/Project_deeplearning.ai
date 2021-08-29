# -*- coding: utf-8 -*-
# @Time    : 2021/8/23 20:17
# @Author  : Ywj
# @File    : gru.py
# @Description : 利用GRU算法对IMDB 进行文本分类

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

MAX_WORDS = 20000  # 词汇表大小
MAX_LEN = 250      # max length
BATCH_SIZE = 50
learning_rate = 0.001
num_epochs = 51
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 借助Keras加载imdb数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
print(x_train.shape, x_test.shape)

# 转化为TensorDataset
train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

# 转化为 DataLoader
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=True)

test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE, drop_last=True)


# 定义gru模型用于文本分类

class GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, context_dim, num_classes):
        super(GRU, self).__init__()
        self.context_dim = context_dim
        # set pretrained glove embeddings for use.
        # freeze the embedding
        self.max_words = input_dim
        self.emb_dim = emb_dim
        self.Embedding = nn.Embedding(self.max_words, self.emb_dim)
        self.gru = nn.GRU(self.emb_dim, context_dim, num_layers=2, bias=True, batch_first=True)
        self.linear = nn.Linear(context_dim, num_classes)

    def forward(self, input, hidden):
        # use given hidden for initial_hidden_states
        # print(input.shape)
        input = self.Embedding(input)
        # print(input.shape)
        all_h, last_h = self.gru(input, hidden)
        # since we have only 1 layer and 1 direction
        output = self.linear(last_h[0])
        # return the last_h to re-feed for next batch
        return output, last_h

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.context_dim).to(DEVICE))


class BiGRU(nn.Module):
    def __init__(self, input_dim, emb_dim, context_dim, drop_out, num_classes):
        super(BiGRU, self).__init__()
        self.context_dim = context_dim
        self.max_words = input_dim
        self.emb_dim = emb_dim
        self.drop_out = drop_out
        self.Embedding = nn.Embedding(self.max_words, self.emb_dim)
        self.gru = nn.GRU(self.emb_dim, context_dim, num_layers=1, bias=True, batch_first=True, dropout=drop_out,
                          bidirectional=True)
        self.dp = nn.Dropout(self.drop_out)
        # since we are using 2 directions
        self.linear1 = nn.Linear(2 * context_dim, context_dim)
        self.linear2 = nn.Linear(context_dim, num_classes)

    def forward(self, input, hidden):
        # we dont need to initialize explicitly -
        # h0 = Variable(torch.zeros(1,input.size(0),self.context_dim))
        input = self.Embedding(input)
        input = self.dp(input)
        all_h, last_h = self.gru(input, hidden)
        last_h = self.dp(last_h)
        # last_h shape is 2,batch_size,context_dim (2 is for 2 directions)
        concated_h = torch.cat([last_h[0], last_h[1]], 1)
        output1 = self.linear1(concated_h)
        output = self.linear2(output1)
        return output, last_h

    def init_hidden(self, batch_size=BATCH_SIZE):
        # since we are using bi-directional use 2 layers.
        return Variable(torch.zeros(2, batch_size, self.context_dim).to(DEVICE))



def train(model, device, train_loader, optimizer, epoch):   # 训练模型
    model.train()
    criterion = nn.CrossEntropyLoss()
    hidden = model.init_hidden()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        hidden.detach_()
        y_, hidden = model(x, hidden)
        loss = criterion(y_, y)  # 得到loss
        loss.backward()
        optimizer.step()
        if(batch_idx + 1) % 10 == 0:    # 打印loss
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):    # 测试模型
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')  # 累加loss
    test_loss = 0.0
    acc = 0
    hidden = model.init_hidden()
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_, hidden = model(x, hidden)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)


input_dim = MAX_WORDS
emb_dim = 100  # embedding dimension
context_dim = 50
num_classes = 2
drop_out = 0.1
model = BiGRU(input_dim, emb_dim, context_dim, drop_out, num_classes).to(DEVICE)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_acc = 0.0
PATH = 'imdb_model/gru_model.pth'  # 定义模型保存路径

for epoch in range(1, num_epochs):  # 50个epoch
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
