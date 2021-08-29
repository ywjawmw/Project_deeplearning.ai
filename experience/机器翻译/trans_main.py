# -*- coding: utf-8 -*-
# @Time    : 2021/8/25 20:06
# @Author  : Ywj
# @File    : trans_main.py
# @Description :  训练机器翻译模型(transformer_based)
import torch
from transformer_seq2seq import *
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def save_model(model, store_model_path,step):
    torch.save(model.state_dict(), f'{store_model_path}/trans_model_{step}.ckpt')
    return


def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    return model


def build_model(config, en_vocab_size, cn_vocab_size):
    # 建构模型
    # hid_size, src_vocab_size, n_layers, n_heads, pf_dim, device, dropout=0.1
    encoder = Encoder(config.hid_size, en_vocab_size, config.n_layers, config.n_heads, config.pf_dim, device, config.dropout).to(device)
    # hid_size, trg_vocab_size, n_layers, n_heads, pf_dim, device, dropout = 0.1
    decoder = Decoder(config.hid_size, cn_vocab_size, config.n_layers, config.n_heads, config.pf_dim, device, config.dropout).to(device)
    model = Seq2Seq(encoder, decoder, device)
    # 建构 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(device)

    return model, optimizer


def toekens2sentence(outputs,int2word):
    '''
    :param outputs: [batch size ,sentence len]
    :param int2word:
    :return:
    '''
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            # print("token.shape: ", token.shape)
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


def computebleu(sentences, targets):
    smooth = SmoothingFunction()
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        '''把词语拆成单字'''
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                # 数字  字母不用拆
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        # print("[target]: ", [target])
        # print("sentence: ", sentence)
        score += sentence_bleu([target], sentence, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=smooth.method1)

    return score


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps,
          teacher_forcing_ratio=None):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        # print("targets.shape: ", targets.shape)
        outputs, preds = model(sources, targets, teacher_forcing_ratio)
        # outputs = [batch size, sentence size, vocab size]
        # preds = [batch size,sentence size]
        # targets 的第一个token 是<BOS>，所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        # outputs = [batch size*sentence size,vocab size]
        targets = targets[:, 1:].reshape(-1,)
        # targets = [ batch size*sentence size ]
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        # 防止梯度爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print('\r', 'train[{}] loss: {:.3f}     '.format(total_steps + step + 1, loss_sum, np.exp(loss_sum)))
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def test(model, dataloader, loss_function):
    model.eval()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []
    for sources, targets in dataloader:
        # print("len(dataloader): ", len(dataloader))
        sources, targets = sources.to(device), targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets)
        # 裁掉第一个<BOS>字符
        # print("outputs.shape: ", outputs.shape)
        outputs = outputs[:, 1:, ].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        # print("outputs.shape: ", outputs.shape)
        # print("preds.shape: ", preds.shape)
        loss = loss_function(outputs, targets)
        loss_sum += loss.item()

        # 把预测结果转成文字
        # targets = [batch size,sentence size,vocab size]
        targets = targets.view(batch_size, -1)  # [batch size,vocab size]---str(int(vocab size))可以变成翻译词典的键
        # 翻译结果
        preds = toekens2sentence(preds, dataloader.dataset.int2word_cn)
        # preds = [batch size,sentence size]
        # 要翻译的句子
        sources = toekens2sentence(sources, dataloader.dataset.int2word_en)
        # 标准答案
        targets = toekens2sentence(targets, dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 计算bleu score # 这一个批次所有句子的分数和
        bleu_score += computebleu(preds, targets)
        n += batch_size
    return loss_sum / len(dataloader), bleu_score / n, result


def train_process(config):
    # 准备训练资料
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 准备检验资料
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=2)  # 这里不能设置成1 torch 会自动降维会出错
    # 建构模型
    print("train_dataset.en_vocab_size: ", train_dataset.en_vocab_size)
    print("train_dataset.cn_vocab_size: ", train_dataset.cn_vocab_size)
    model, optmizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=1)

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < config.num_steps):
        # 训练模型
        model, optmizer, loss = train(model, optmizer, train_iter, loss_function, total_steps, config.summary_steps,teacher_forcing_ratio=config.teacher_forcing_ratio)
        train_losses += loss
        # 检验模型
        val_loss, bleu_score, result = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        total_steps += config.summary_steps
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.5f}       ".format(total_steps, val_loss,
                                                                                                  np.exp(val_loss),
                                                                                                  bleu_score))

        # 存储模型和结果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, config.store_model_path, total_steps)

    return train_losses, val_losses, bleu_scores, result


class Configurations:
    def __init__(self):
        self.hid_size = 256
        self.n_heads = 8
        self.n_layers = 3
        self.pf_dim = 512
        self.dropout = 0.1
        self.learning_rate = 0.0005
        self.batch_size = 128
        # self.emb_dim = 256
        # self.dec_hid_dim = 512
        # self.enc_hid_dim = 512

        self.max_output_len = 50
        self.num_steps = 12000  # 总训练次数
        self.store_steps = 300  # 训练多少次后存储模型
        self.summary_steps = 300  # 训练多少次测试一次，查看是否过拟合
        self.load_model = False
        self.store_model_path = 'model'
        self.load_model_path = None
        self.data_path = '/home/ywj/Myproject/Project_deeplearning.ai/experience/数据集/机器翻译CMN-ENG/cmn-eng'
        self.attention = True
        self.teacher_forcing_ratio = 0.5


if __name__ == '__main__':
  config = Configurations()
  print ('config:\n', vars(config))
  train_losses, val_losses, bleu_scores, result = train_process(config)
  print(result[85])
