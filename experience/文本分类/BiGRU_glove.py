# -*- coding: utf-8 -*-
# @Time    : 2021/8/24 20:53
# @Author  : Ywj
# @File    : BiGRU.py
# @Description : 使用双向GRU进行文本分类

import os
from  tqdm import tqdm
from collections import Counter
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
from torchtext.vocab import Vocab
import itertools
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

imdb_dir = '/home/ywj/Myproject/Project_deeplearning.ai/experience/数据集/文本分类IMDB/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# 载入训练集和测试集中的neg（消极）和 pos（积极）数据文本并打上标签
def read_test_train_dir(path):
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(path, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
    return texts, labels


train_texts,train_labels = read_test_train_dir(train_dir)
test_texts, test_labels = read_test_train_dir(test_dir)
print("one sample for train & test text and label:")
print(train_texts[:1], " <--> ", train_labels[:1], "len: ", len(train_texts))
print(test_texts[:1], " <--> ", test_labels[:1],  "len: ", len(test_texts))


def get_paragraph_words(text):
    return flatten([word_tokenize(s) for s in sent_tokenize(text)])


sent_tokenize = nltk.sent_tokenize
word_tokenize = RegexpTokenizer(r'\w+').tokenize


def word_tokenize_para(text):
    return [word_tokenize(s) for s in sent_tokenize(text)]


def flatten(l):
    return [item for sublist in l for item in sublist]


vocab_counter = Counter(flatten([get_paragraph_words(text) for text in train_texts]))
# print(vocab_counter)

# 词汇表的glove词向量
w2v = Vocab(vocab_counter, max_size=20000, min_freq=3, vectors='glove.6B.100d')
# randomly shuffle the training data
training_set = list(zip(train_texts, train_labels))
# shuffle works inplace and returns None .
random.shuffle(training_set)

# randomly shuffle the training data
testing_set = list(zip(test_texts, test_labels))
# shuffle works inplace and returns None .
random.shuffle(testing_set)
maxSeqLength = 250


# 从文本返回索引列表中获取词汇索引的函数（在maxSeqLength处截断）
def stoiForReview(w2v, text, maxSeqLength):
    # 将句子修剪为maxSeqLength，否则返回原始长度。
    return [w2v.stoi[word] for word in get_paragraph_words(text)[0:maxSeqLength]]


# review需要的词向量 - returns tensor of size (1, min(len(review),maxSeqLength), embedded_dim)
def wordVectorsForReview(w2v, text, maxSeqLength):
    indexes = stoiForReview(w2v, text, maxSeqLength)
    # returns tensor with size [num_words,1,embedding_dim]
    # That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.
    sent_word_vectors = torch.cat([w2v.vectors[i].view(1, -1) for i in indexes]).view(len(indexes), 1, -1)

    # batch first (1,seq_len,embedding_dim)
    # seq_len has been maximized to maxSeqLength
    sent_word_vectors = sent_word_vectors.view(1, len(sent_word_vectors), -1)

    return sent_word_vectors


# Create nn.Emedding for easy lookup
idx2vec = w2v.vectors
embedding = nn.Embedding(idx2vec.shape[0], idx2vec.shape[1])
embedding.weight = nn.Parameter(idx2vec)
embedding.weight.requires_grad = False


def get_batch(t_set, str_idx, end_idx):
    training_batch_set = t_set[str_idx:end_idx]

    input_texts, labels = zip(*training_batch_set)

    # convert texts to vectors shape - Batch(=1),seq_length(cut-off at maxSeqLength),embedded_dim
    input_vectors = [wordVectorsForReview(w2v, text, maxSeqLength) for text in input_texts]

    # convert to variable w/ long tensor
    labels = Variable(torch.LongTensor(labels))

    seq_lens = torch.LongTensor([i.shape[1] for i in input_vectors])
    embedding_dim = input_vectors[0].shape[2]
    # batch_inputs  - [batch_size, seq_len,embedding_dim]
    # print("embedding_dim:", embedding_dim)
    batch_inputs = Variable(torch.zeros((len(seq_lens), seq_lens.max(), embedding_dim)))
    for idx, (seq, seqlen) in enumerate(zip(input_vectors, seq_lens)):
        batch_inputs[idx, :seqlen] = seq
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    batch_inputs = batch_inputs[perm_idx]
    batch_inputs = pack_padded_sequence(batch_inputs, seq_lens.numpy(), batch_first=True)
    labels = labels[perm_idx]
    return batch_inputs.to(device), labels.to(device)


# def repackage_hidden(h):
#     """Wraps hidden states in new Variables, to detach them from their history."""
#     if type(h) == Variable:
#         return Variable(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)


class GRU(nn.Module):
    def __init__(self, input_dim, context_dim, num_classes):
        super(GRU, self).__init__()
        self.context_dim = context_dim
        # set pretrained glove embeddings for use.
        # freeze the embeddin

        self.gru = nn.GRU(input_dim, context_dim, 2, bias=True, batch_first=True)
        self.linear = nn.Linear(context_dim, num_classes)

    def forward(self, input, hidden):
        # use given hidden for initial_hidden_states
        all_h, last_h = self.gru(input, hidden)
        # since we have only 1 layer and 1 direction
        output = self.linear(last_h[0])
        # return the last_h to re-feed for next batch
        return output, last_h

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.context_dim).to(device))


class BiGRU(nn.Module):
    def __init__(self, input_dim, context_dim, num_classes):
        super(BiGRU, self).__init__()
        self.context_dim = context_dim
        self.gru = nn.GRU(input_dim, context_dim, num_layers=2, bias=True, batch_first=True, dropout=0,
                          bidirectional=True)
        # since we are using 2 directions
        self.linear = nn.Linear(2 * context_dim, num_classes)

    def forward(self, input, hidden):
        # we dont need to initialize explicitly -
        # h0 = Variable(torch.zeros(1,input.size(0),self.context_dim))
        all_h, last_h = self.gru(input, hidden)
        # last_h shape is 2,batch_size,context_dim (2 is for 2 directions)
        concated_h = torch.cat([last_h[0], last_h[1]], 1)
        output = self.linear(concated_h)
        return output, last_h

    def init_hidden(self, batch_size):
        # since we are using bi-directional use 2 layers.
        return Variable(torch.zeros(4, batch_size, self.context_dim).to(device))


# 定义lstm模型用于文本分类
class LSTM(nn.Module):
    def __init__(self, emb_size, hid_size, dropout=0.1):
        super(LSTM, self).__init__()
        # self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        # self.Embedding = nn.Embedding(self.max_words, self.emb_size)
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
        # x = self.Embedding(x)  # [bs, ml, emb_size]
        # print(x.shape)
        # x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]


learning_rate = 0.001
batch_size = 50
num_passes = 25000//batch_size  # number of batches with given batch_size
num_epochs = 50  # number of times we will go through all the training samples
input_dim = 100  # embedding dimension
context_dim = 50
num_classes = 2

criterion = nn.CrossEntropyLoss()
model = BiGRU(input_dim, context_dim, num_classes).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()
for epoch in range(num_epochs):
    #re-initialize after
   # random.shuffle(training_set)
    hidden = model.init_hidden(batch_size)
    # reinitialize hidden layers to zero after each epoch
    for i in range(num_passes):
        str_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        inputs, labels = get_batch(training_set, str_idx, end_idx)
        # print("type(inputs): ", type(inputs))
        # print("inputs.shape: ", inputs.shape)
        # print("inputs[0].shape: ", inputs[0].shape)
        hidden.detach_()
        optimizer.zero_grad()  # zero the gradient buffer
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            # print(epoch)
            print('pass [%d/%d], in epoch [%d/%d] Loss: %.4f'
                  % (i + 1, num_passes, epoch, num_epochs, loss.item()))
model.eval()
correct = 0
total = 0
hidden = model.init_hidden(batch_size)
for i in range(num_passes):
    str_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    testing_inputs, testing_labels = get_batch(training_set, str_idx, end_idx)
    outputs, hidden = model(testing_inputs, hidden)
    _, predicted = torch.max(outputs.data, 1)
    total += testing_labels.size(0)
    correct += (predicted == testing_labels.data).sum()
print('Accuracy of the network on the  training data : %d %%' % (100 * correct / total))

# Test
model.eval()
correct = 0
total = 0
hidden = model.init_hidden(batch_size)
for i in range(num_passes):
    str_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    testing_inputs, testing_labels = get_batch(testing_set, str_idx, end_idx)
    outputs, hidden = model(testing_inputs, hidden)
    _, predicted = torch.max(outputs.data, 1)
    total = testing_labels.size(0)
    correct = (predicted == testing_labels.data).sum()
print('Accuracy of the network on the  test data: %d %%' % (100 * correct / total))



# def train(epoch, num_epochs, batch_size, num_passes):
#     hidden = model.init_hidden(batch_size)
#     # reinitialize hidden layers to zero after each epoch
#     for i in range(num_passes):
#         str_idx = i * batch_size
#         end_idx = (i + 1) * batch_size
#         inputs, labels = get_batch(training_set, str_idx, end_idx)
#         # print("type(inputs): ", type(inputs))
#         # print("inputs[0].shape: ", inputs[0].shape)
#         hidden.detach_()
#         optimizer.zero_grad()  # zero the gradient buffer
#         outputs, hidden = model(inputs, hidden)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 100 == 0:
#             print('pass [%d/%d], in epoch [%d/%d] Loss: %.4f'
#                   % (i + 1, num_passes, epoch, num_epochs, loss.item()))
#
#
# def evl(batch_size, num_passes):
#     model.eval()
#     correct = 0
#     total = 0
#     hidden = model.init_hidden(batch_size)
#     for i in range(num_passes):
#         str_idx = i * batch_size
#         end_idx = (i + 1) * batch_size
#         testing_inputs, testing_labels = get_batch(training_set, str_idx, end_idx)
#         outputs, hidden = model(testing_inputs, hidden)
#         _, predicted = torch.max(outputs.data, 1)
#         total += testing_labels.size(0)
#         correct += (predicted == testing_labels.data).sum()
#     print('Accuracy of the network on the  training data : %d %%' % (100 * correct / total))
#
#     # testing_inputs, testing_labels = get_batch(training_set, 0, 25000)
#     # hidden = model.init_hidden(25000)
#     # outputs, hidden = model(testing_inputs, hidden)
#     # _, predicted = torch.max(outputs.data, 1)
#     # total = testing_labels.size(0)
#     # correct = (predicted == testing_labels.data).sum()
#     # print('Accuracy of the network on the  training data : %d %%' % (100 * correct / total))
#
#
# def test(batch_size, num_passes):
#     # Test the Model on training data
#     model.eval()
#     correct = 0
#     total = 0
#     hidden = model.init_hidden(batch_size)
#     for i in range(num_passes):
#         str_idx = i * batch_size
#         end_idx = (i + 1) * batch_size
#         testing_inputs, testing_labels = get_batch(testing_set, str_idx, end_idx)
#         outputs, hidden = model(testing_inputs, hidden)
#         _, predicted = torch.max(outputs.data, 1)
#         total = testing_labels.size(0)
#         correct = (predicted == testing_labels.data).sum()
#     print('Accuracy of the network on the  test data: %d %%' % (100 * correct / total))




