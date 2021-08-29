# -*- coding: utf-8 -*-
# @Time    : 2021/8/25 19:53
# @Author  : Ywj
# @File    : transformer_seq2seq.py
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
import math

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


# 位置编码
class Position_wise(nn.Module):
    def __init__(self, device):
        super(Position_wise, self).__init__()
        self.device = device

    def forward(self, src):
        # src[batch src_len hid_size]
        batch = src.shape[0]
        src_len = src.shape[1]
        d_model = src.shape[2]    # hidden size

        pos_embedd = torch.zeros_like(src).to(device)
        # pos_embedd[batch src_len d_model]

        pos = torch.arange(0, src_len).unsqueeze(0).unsqueeze(-1).to(self.device)
        # pos[1 src_len 1]

        pos = pos.repeat(batch, 1, int(d_model / 2))
        # pos[batch src_len  d_model/2]
        div = torch.exp(torch.arange(0, d_model / 2) * (-math.log(10000.0) / d_model)).to(device)
        # div[d_model/2]
        pos_embedd[:, :, 0::2] = torch.sin(pos * div)
        pos_embedd[:, :, 1::2] = torch.cos(pos * div)
        return pos_embedd


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_size, n_heads, device, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # 三个线性层模板
        self.Q = nn.Linear(hid_size, hid_size)
        self.K = nn.Linear(hid_size, hid_size)
        self.V = nn.Linear(hid_size, hid_size)

        # 多头拼接层
        self.fc = nn.Linear(hid_size, hid_size)

        # 各种参数记录
        self.hid_size = hid_size
        self.n_heads = n_heads
        self.heads_dim = hid_size // n_heads
        self.scale = torch.sqrt(torch.FloatTensor([self.heads_dim])).to(device)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, q, k, v, masked=None):
        # 首先经历三个线性变化得到q,v,k向量
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        batch = q.shape[0]

        # 由于是多头自注意力，我们将维度hid_size分成n_heads份
        # 每一个多头我们希望其关注不同侧重点
        q = q.reshape(batch, -1, self.n_heads, self.heads_dim)
        # q[batch seq_len n_heads heads_dim]
        q = q.permute(0, 2, 1, 3)
        # q[batch n_heads seq_len heads_dim]
        k = k.reshape(batch, -1, self.n_heads, self.heads_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, -1, self.n_heads, self.heads_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        # energy[batch n_head seq_len1 seq_len]

        # 将energy通进行mask忽视pad
        if masked is not None:
            energy = energy.masked_fill(masked == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention[batch n_head seq_len1 seq_len]

        # 对权重与值向量加权求和得到上下文向量
        context = torch.matmul(self.dropout(attention), v)
        # context[batch n_head seq_len1 heads_dim]

        # 拼接各个头并进行维度变化输出
        context = context.permute(0, 2, 1, 3).reshape(batch, -1, self.hid_size)
        # context[batch seq_len hid_size]
        output = self.fc(context)
        return output, attention


class FeedfordNN(nn.Module):
    def __init__(self, hid_size, pf_dim, dropout=0.1):
        super(FeedfordNN, self).__init__()
        # hid_size表示嵌入层  隐藏的维度，
        # pf_dim表示升维的维度
        self.fc1 = nn.Linear(hid_size, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.dropout(torch.relu(self.fc1(src)))
        src = self.fc2(src)
        return src


class Transform(nn.Module):
    def __init__(self, hid_size, n_heads, pf_dim, device, dropout=0.1):
        super(Transform, self).__init__()
        self.self_attention = MultiHeadAttention(hid_size, n_heads, device)
        self.self_attention_layer_norm = nn.LayerNorm(hid_size)
        self.feedforward = FeedfordNN(hid_size, pf_dim)
        self.feedforward_layer_norm = nn.LayerNorm(hid_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_masked):
        # src为带位置编码的词嵌入
        # src[batch src_len  hid_size]

        # 经过多头自自注意力
        _src, _ = self.self_attention(src, src, src, src_masked)
        # 进行残差连接并层归一化
        # src[batch src_len  hid_size]
        src = self.self_attention_layer_norm(src + self.dropout(_src))
        # src[batch src_len  hid_size]
        # 经过前馈神经网络层
        _src = self.feedforward(src)
        # 进行残差连接并进行层归一化
        src = self.feedforward_layer_norm(src + self.dropout(_src))
        return src


class Encoder(nn.Module):
    def __init__(self, hid_size, src_vocab_size, n_layers, n_heads, pf_dim, device, dropout=0.1):
        super(Encoder, self).__init__()
        # hid_size：隐层维度与嵌入层维度
        # src_vocab_size：词库大小
        # n_layers：transformer的层结构
        # n_heads:注意力头的数量
        # pf_dim:前馈层上升的维度

        self.token_emb = nn.Embedding(src_vocab_size, hid_size, padding_idx=1)
        self.pos_emb = Position_wise(device)
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            Transform(hid_size, n_heads, pf_dim, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 主归一化功能
        self.scale = torch.sqrt(torch.FloatTensor([hid_size])).to(device)

    def forward(self, src):
        # print("encoder input shape pre : ", src.shape)
        # src = src[:, 0:2]
        # print("encoder input shape after : ", src.shape)
        # 获取掩码编码
        src_masked = (src != 1)
        # src[batch src_len]
        src_masked = src_masked.unsqueeze(1).unsqueeze(1)
        # src[batch 1 1 src_len]

        # 对输入的源句嵌入
        # src[batch src_len]
        # print("encoder input shape pre : ", src.shape)
        src = self.token_emb(src)
        # print("encoder embedding input shape : ", src.shape)
        # src[batch src_len hid_size]

        # 词嵌入编码
        pos = self.pos_emb(src)
        # pos[batch src_len hid_size]
        intput_transformer = self.dropout(src * self.scale + pos)
        # intput_transformer[batch src_len hid_size]

        for transform in self.layers:
            intput_transformer = transform(intput_transformer, src_masked)

        # intput_transformer[batch src_len hid_size]
        return intput_transformer, src_masked


class De_Transformer(nn.Module):
    def __init__(self, hid_size, n_heads, pf_dim, device, dropout=0.1):
        super(De_Transformer, self).__init__()
        self.device = device
        self.masked_mutil_attention = MultiHeadAttention(hid_size, n_heads, device)
        self.masked_mutil_attention_layer_norm = nn.LayerNorm(hid_size)
        self.mutil_attention = MultiHeadAttention(hid_size, n_heads, device)
        self.mutil_attention_layer_norm = nn.LayerNorm(hid_size)
        self.feedNN = FeedfordNN(hid_size, pf_dim)
        self.feedNN_layer_norm = nn.LayerNorm(hid_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_outputs, trg, trg_pad_mask, enc_masked):
        # enc_outputs encoder层的输出[batch src_len hide_size]
        # trg      带有位置信息的目标句[batch trg_len hid_size]
        # enc_masked 源句的掩码信息[batch 1 1 src_len]
        # trg_pad_mask[batch 1 1 trg_len]

        trg_len = trg.shape[1]
        trg_mask_ans = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_mask_ans [trg_len trg_len]

        trg_mask = trg_pad_mask & trg_mask_ans
        # trg_mask_ans[batch 1 trg_len trg_len]

        _trg, _ = self.masked_mutil_attention(trg, trg, trg, trg_mask)
        trg = self.masked_mutil_attention_layer_norm(trg + self.dropout(_trg))
        # trg[batch trg_len hid_dize]

        _trg, attention = self.mutil_attention(trg, enc_outputs, enc_outputs, enc_masked)
        trg = self.mutil_attention_layer_norm(trg + self.dropout(_trg))
        # trg[batch trg_len hid_dize]

        _trg = self.feedNN(trg)
        trg = self.feedNN_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, hid_size, trg_vocab_size, n_layers, n_heads, pf_dim, device, dropout=0.1):
        super(Decoder, self).__init__()
        # hid_size：隐层维度与嵌入层维度
        # src_vocab_size：德语词库大小
        # n_layers：transformer的层结构

        self.token_emb = nn.Embedding(trg_vocab_size, hid_size, padding_idx=1)
        self.pos_emb = Position_wise(device)
        self.n_layers = n_layers
        self.trg_vocab_size = trg_vocab_size

        self.layers = nn.ModuleList([
            De_Transformer(hid_size, n_heads, pf_dim, device)  # Decoder的Transformer抽象实现
            for _ in range(n_layers)
        ])

        # 分类
        self.fc = nn.Linear(hid_size, trg_vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 主归一化功能
        self.scale = torch.sqrt(torch.FloatTensor([hid_size])).to(device)

    def forward(self, trg, enc_outputs, enc_mask):
        # trg[batch trg_len]
        # enc_outputs[batch src_len hid_size]
        # mask[batch 1 1 src_len]
        # trg[batch src_len vocab_size]
        # print("decoder input shape pre : ", trg.shape)
        # trg = trg[:, 0:2]
        # print("decoder input shape after : ", trg.shape)
        # 目标的pad_mask
        pad_mask = (trg != 1).unsqueeze(1).unsqueeze(1)
        # [batch trg_len]
        # 对输入的目标句嵌入
        # trg[batch trg_len]
        # print("decoder input shape pre : ", trg.shape)
        trg = self.token_emb(trg)
        # print("decoder embedding input shape : ", trg.shape)
        # trg[batch trg_len hid_size]

        # 对词嵌入嵌入编码信息
        pos = self.pos_emb(trg)
        # pos[batch trg_len hid_size]
        intput_transformer = self.dropout(trg * self.scale + pos)
        # intput_transformer[batch trg_len hid_size]

        for transform in self.layers:
            intput_transformer, _ = transform(enc_outputs, intput_transformer, pad_mask, enc_mask)
        # intput_transformer[batch trg_len hid_size]

        output = self.fc(intput_transformer)
        # output[batch trg_len trg_vocab_size]
        return output


# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, src, trg):
#         # src[batch src_len]
#         # trg[batch src_len]
#         enc_outputs, enc_mask = self.encoder(src)
#         outputs = self.decoder(trg, enc_outputs, enc_mask)
#         return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            'encoder and decoder must have equal number of layers'

    def forward(self, input, target, teacher_forcing_ratio):
        # input = [batch size, input len]
        # target = [batch size,target len]
        # teacher_forcing_ratio 有多少几率使用正确答案来训练
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.trg_vocab_size

        # 准备一个存储空间来存翻译的结果
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 将输入放入Encoder
        encoder_outputs, enc_mask = self.encoder(input)
        # 取出<BOS> token
        input = target
        preds = []
        # output[batch trg_len trg_vocab_size]
        output = self.decoder(input, encoder_outputs, enc_mask)
        outputs = output
        for t in range(1, target_len):
            # 决定是否用正确答案做训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出几率最大的单词
            top1 = output[:, t].argmax(1)
            # 如果是teacher force 用正解训练，反之用自己预测的单词预测
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        # preds = torch.cat(preds, 1)
        # # 决定是否用正确答案做训练
        # teacher_force = random.random() <= teacher_forcing_ratio
        # # 取出几率最大的单词
        # top1 = output.argmax(1)
        # # 如果是teacher force 用正解训练，反之用自己预测的单词预测
        # input = target if teacher_force else top1
        # preds.append(top1.unsqueeze(1))
        # for t in range(1, target_len):
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target):
        '''
        test
        :param input:
        :param target:
        :return:
        '''
        # input = [batch size, input len]
        # target = [batch size,target len]
        # teacher_forcing_ratio 有多少几率使用正确答案来训练
        # print("--------")
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.trg_vocab_size

        # 准备一个存储空间来存翻译的结果
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 将输入放入Encoder
        encoder_outputs, enc_mask = self.encoder(input)
        # 取出<BOS> token
        input = target
        preds = []
        output = self.decoder(input, encoder_outputs, enc_mask)
        # print("output.shape: ", output.shape)
        # # output[batch size, vocab size]
        # # print("output.shape::", output.shape)
        # outputs = output
        # # 取出几率最大的单词
        # top1 = output.argmax(1)  # [batch size]
        # # top1 = output.argmax(0)  # [batch size]这里batch size 是一 给自动降维了?
        # input = top1
        # preds.append(top1.unsqueeze(1))  # [bath size,1]
        # # for t in range(1, target_len):
        # # print("--preds.shape: ", preds[0].shape)
        # preds = torch.cat(preds, 1)  # [batch size,translated words size]
        outputs = output
        for t in range(1, target_len):
            # 取出几率最大的单词
            top1 = output[:, t].argmax(1)  # [batch size]
            # top1 = output.argmax(0)  # [batch size]这里batch size 是一 给自动降维了?
            preds.append(top1.unsqueeze(1))  # [bath size,1]
        preds = torch.cat(preds, 1)  # [batch size,translated words size]
        # print("--preds.shape: ", preds.shape)
        return outputs, preds