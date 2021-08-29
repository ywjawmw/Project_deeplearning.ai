# -*- coding: utf-8 -*-
# @Time    : 2021/8/23 20:17
# @Author  : Ywj
# @File    : lstm.py
# @Description : 利用LSTM算法对IMDB 进行文本分类

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from torch.autograd import Variable

MAX_WORDS = 10000  # 词汇表大小
MAX_LEN = 250      # max length
# MAX_LEN = 16
BATCH_SIZE = 250
# EMB_SIZE = 128   # embedding size
# HID_SIZE = 128   # lstm hidden size
EMB_SIZE = 128
HID_SIZE = 128
DROPOUT = 0.1
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 借助Keras加载imdb数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
# x_train, x_test : (len(train_set), MAX_LEN) (len(test_set), MAX_LEN)
print(x_train.shape, x_test.shape)

# 转化为TensorDataset
train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

# 转化为 DataLoader
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


# 定义lstm模型用于文本分类
class Model(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout):
        super(Model, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=True)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        """
        input : [bs, maxlen]
        output: [bs, 2]
        """
        # print(x.shape)
        x = self.Embedding(x)  # [bs, ml, emb_size]
        # print(x.shape)
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]


class BI_LSTM_Attention(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, use_cuda, attention_size, sequence_length):
        super(BI_LSTM_Attention, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.lookup_table.weight.data.uniform_(-1., 1.)
        self.dp = nn.Dropout(self.dropout)

        self.layer_size = 2
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            num_layers=self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            )

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        if self.use_cuda:
            self.w_omega = Variable(torch.zeros(self.hidden_size * 2, self.attention_size).to(DEVICE))
            self.u_omega = Variable(torch.zeros(self.attention_size).to(DEVICE))
        else:
            self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
            self.u_omega = Variable(torch.zeros(self.attention_size))
        # self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.label = nn.Linear(self.hidden_size, self.output_size)
        self.Bi_label = nn.Linear(hidden_size * 2, output_size)


    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # lstm_output (sequence_length, batch_size, hidden_size*layer_size)
        # print(lstm_output.shape)
        # print(self.layer_size)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * 2])
        # output_reshape (sequence_length * batch_size, hidden_size*layer_size)

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # attn_tanh (sequence_length * batch_size, attention_size)

        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # attn_hidden_layer (sequence_length * batch_size, 1)

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # exps (batch_size, sequence_length)

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # alphas (batch_size, sequence_length)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # alphas_reshape (batch_size, sequence_length, 1)

        state = lstm_output.permute(1, 0, 2)
        # state  (batch_size, sequence_length, hidden_size*layer_size)
        # print("state: ", state.shape)
        # print("alphas_reshape", alphas_reshape.shape)

        attn_output = torch.sum(state * alphas_reshape, 1)
        # attn_output  (batch_size, hidden_size*2)

        return attn_output

    def forward(self, input_sentences):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).to(DEVICE))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).to(DEVICE))
        else:
            h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        input = self.dp(input)
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.Bi_label(attn_output)
        # without attention
        # input = self.dp(input)
        # lstm_output, _ = self.lstm(input)
        # lstm_output = self.dp(lstm_output)
        # x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        # x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        # logits = self.label(x)
        return logits


def train(model, device, train_loader, optimizer, epoch):   # 训练模型
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_ = model(x)
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
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x)
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]   # .max() 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)


lr = 0.001
weight_decay = 0.001
attention_size = 16
sequence_length = 16
# model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
model = BI_LSTM_Attention(BATCH_SIZE, 2, HID_SIZE, MAX_WORDS, EMB_SIZE, True, DROPOUT, True, attention_size, MAX_LEN).to(DEVICE)
print(model)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

best_acc = 0.0
PATH = 'imdb_model/lstm_model1.pth'  # 定义模型保存路径

for epoch in range(1, 41):  # 30个epoch
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
