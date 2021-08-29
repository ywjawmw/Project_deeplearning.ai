# -*- coding: utf-8 -*-
# @Time    : 2021/8/25 19:53
# @Author  : Ywj
# @File    : attention_seq2seq.py
# @Description : 基于attention的transformer方法进行机器翻译

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import sampler
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import random
import re
import json

if not os.path.exists('model'):
    os.mkdir('model')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# 把句子补成相同长度
class LabelTransformer:
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, self.size - label.shape[0]), mode='constant', constant_values=self.pad)
        return label


# 处理数据集CMN-ENG
class EN2CNDataset(data.Dataset):
    def __init__(self, root, max_output_len, set_name):
        self.root = root
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # 载入资料
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), 'r') as F:
            for line in F:
                self.data.append(line)
        print(f'{set_name} dataset size : {len(self.data)}')

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)
        self.transform = LabelTransformer(max_output_len, self.word2int_en['<PAD>'])

    def get_dictionary(self, language):
        '''
        :param language: en 或者cn
        :return: 对应语言的id转文字和文字转id 的字典
        '''
        # 载入字典
        with open(os.path.join(self.root, f'word2int_{language}.json'), 'r') as F:
            word2int = json.load(F)

        with open(os.path.join(self.root, f'int2word_{language}.json'), 'r') as F:
            int2word = json.load(F)

        return word2int, int2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # 先把中英文分开
        sentences = self.data[item]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        assert len(sentences) == 2

        # 准备特殊字元
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在开头添加<BOS>,在结尾添加<EOS>,不在字典的subword(词) 用<UNK>代替
        en, cn = [BOS], [BOS]
        # 将句子拆解为subword 并转为整数
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)
        en, cn = np.asarray(en), np.asarray(cn)
        # 用<PAD>将句子补到相同长度
        en, cn = self.transform(en), self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn


class Encode(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        self.embeding = nn.Embedding(en_vocab_size, emb_dim)
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim*2,dec_hid_dim)

    def forward(self, input):
        # input = [batch size ,sequence len,vocab size]
        # print("input: ", input.shape)
        embeding = self.embeding(input)
        outputs, hidden = self.rnn(self.dropout(embeding))
        # outputs = [batch size,sequence len,hid dim * directions]
        # hidden = [numlayer*directions,batch size ,hid dim ]
        # outputs 是最上一层的rnn 的输出 directions是因为这里用了双向注意力
        batch_size = outputs.shape[0]
        s = hidden.view(self.n_layers, 2, batch_size, -1)
        # [layers,derections,batch_size,enc_hid_dim]
        s = torch.cat((s[-1, -2, :, :], s[-1, -1, :, :]), dim=1)
        s = torch.tanh(self.fc(s)) # [batch size,dec_hid_dim]
        return outputs, s, hidden

class Attention(nn.Module):
    '''

    1. repeat :  s [batch size,dec_hid_dim] --> [batch size,sentence length,dec_hid_dim]
    2. 拼接s和enc_output并送进全连接网络 nn.Linear(enc_hid_dim*2+dec_hid_dim,dec_hid_dim,bias=False)
    3. 再送入一个全连接网络 nn.Linear(dec_hid_dim,1,bias=False)
    4. 再送入soft max 得到注意力
    '''

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        '''
        :param hidden: the hidden of encoder  # hidden = [batch size, n layers * directions, hid dim]
        :param enc_output: outputs of encoder
        :param max_output_len: sentence length
        :return:
        '''
        # s [ batch size, dec_hid_dim]
        # enc_output [batch size,sentence length,enc_hid_dim*2]
        batch_size = enc_output.shape[0]
        sentence_len = enc_output.shape[1]
        s_new = s.unsqueeze(1).repeat(1, sentence_len, 1)
        # s_new[ batch size, sentence_len,dec_hid_dim]
        # enc_output[batch size, sequence len, hid dim * directions]
        energy = torch.tanh(self.attn(torch.cat((s_new, enc_output), dim=2)))
        # energy [batch size ,snetence len ,dec_hid_dim)

        attention = self.v(energy).squeeze()
        # attention [batch size ,sentence len]
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, cn_vocb_size, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, isatt):
        '''

        :param cn_vocb_size: cn_vocab_size
        :param emb_dim: dimesion of embeded vect
        :param enc_hid_dim: dimesion of encoder hidden
        :param dec_hid_dim:
        :param dropout:
        :param isatt: 是否使用attention
        '''
        super().__init__()
        self.cn_vocab_size = cn_vocb_size
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.embeding = nn.Embedding(cn_vocb_size, emb_dim)
        self.input_dim = emb_dim
        self.hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.isatt = isatt
        if isatt:
            self.input_dim = (enc_hid_dim * 2) + emb_dim
        else:
            pass
        self.rnn = nn.GRU(self.input_dim, dec_hid_dim, self.n_layers, batch_first=True,
                          dropout=dropout)  # 这里提前知不知道翻译结果，不能双向注意力流

        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        '''
        :param dec_input: 翻译好的单词结果 [batch size ,vocab size ]
        :s :最上一层的两个方向的影藏层拼接后转化到dec_hid_dim [batch size,dec_hid_dim]
        :param enc_output: encoder的输出 [batch size,sentence size,enc_hid_dim*2]
        :return:
        '''
        dec_input = dec_input.unsqueeze(1)  # [batch size,1,vocab size ]
        embedded = self.dropout(self.embeding(dec_input))  # [batch_size,1, emb_dim]
        if self.isatt:
            a = self.attention(s, enc_output)  # a[batch size ,sentence len]
            a = a.unsqueeze(1)  # [batch size ,1,sentence size]
            c = torch.bmm(a, enc_output)  # [batch size ,1,enc_hid_dim*2]
            rnn_input = torch.cat((c, embedded), dim=2)  # [batch size ,1,enc_hid_dim*2+emb_dim]
        else:
            rnn_input = embedded
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0).repeat(self.n_layers,1,1))
        # dec_output [batch size,1,hidden dim]
        # dec_hidden = [num_layers, batch size, hid dim]
        dec_output = dec_output.squeeze()  # [batch size,hidden dim]
        # dec_hidden = [num_layers, batch size, hid dim]
        dec_output = self.embedding2vocab1(dec_output)
        dec_output = self.embedding2vocab2(dec_output)
        prediction = self.embedding2vocab3(dec_output)
        # prediction = [batch size, vocab size]
        return prediction, dec_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            'encoder and decoder must have equal number of layers'

    def forward(self, input, target, teacher_forcing_ratio):
        # input = [batch size, input len,vocab size]
        # target = [batch size,target len ,vocab size]
        # teacher_forcing_ratio 有多少几率使用正确答案来训练
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 准备一个存储空间来存翻译的结果
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 将输入放入Encoder
        encoder_outputs, s, hidden = self.encoder(input)
        # 取出<BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, s, encoder_outputs)
            outputs[:, t] = output
            # 决定是否用正确答案做训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出几率最大的单词
            top1 = output.argmax(1)
            # 如果是teacher force 用正解训练，反之用自己预测的单词预测
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target):
        '''
        test
        :param input:
        :param target:
        :return:
        '''
        # input = [batch size, input len,vocab size]
        # target = [batch size,target len ,vocab size]
        # teacher_forcing_ratio 有多少几率使用正确答案来训练
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 准备一个存储空间来存翻译的结果
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 将输入放入Encoder
        encoder_outputs, s, hidden = self.encoder(input)
        # 取出<BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, s, encoder_outputs)
            # print("output.shape: ", output.shape)
            # output[batch size, vocab size]
            outputs[:, t] = output
            # 取出几率最大的单词
            top1 = output.argmax(1)  # [batch size]
            # top1 = output.argmax(0)  # [batch size]这里batch size 是一 给自动降维了?
            input = top1
            preds.append(top1.unsqueeze(1))  # [bath size,1]
        # print("outputs.shape: ", outputs.shape)
        # print("--preds.shape: ", preds[0].shape)
        preds = torch.cat(preds, 1)  # [batch size,translated words size]
        return outputs, preds