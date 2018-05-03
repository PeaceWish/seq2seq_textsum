#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 23:21
# @Author  : Zhangyuheng
# @File    : data_utils.py

import numpy as np
import json
import jieba
import re

def pre_data():
    sum_file = open(r"data\summarization.txt", 'w+', encoding='utf-8')
    artcle_file = open(r"data\article.txt", 'w+', encoding='utf-8')
    with open(r'data\evaluation_with_ground_truth.txt', 'rb') as file:
        for line in file.readlines():
            s = json.loads(line)['summarization'] + '\n'
            a = json.loads(line)['article'] + '\n'
    sum_file.close()
    artcle_file.close()

# pre_data()


def clean_data(filename):
    n_file = "data"+"\\"+filename+"_cleaned.txt"
    f = open(n_file, 'w+', encoding="utf-8")
    with open("data"+"\\"+filename+".txt", 'r+', encoding="utf-8") as file:
        for text in file.readlines():
            text = re.sub(r"<Paragraph>", "", text)
            text = re.sub(r"（", "(", text)
            text = re.sub(r"）", ")", text)
            text = re.sub(r"\(.*?\)", "", text)
            text = re.sub(r"□", "", text)
            text = re.sub(r"●", " ", text)
            text = re.sub(r"▶", "", text)
            text = re.sub(u"０", u"0", text)
            text = re.sub(u"１", u"1", text)
            text = re.sub(u"２", u"2", text)
            text = re.sub(u"３", u"3", text)
            text = re.sub(u"４", u"4", text)
            text = re.sub(u"５", u"5", text)
            text = re.sub(u"６", u"6", text)
            text = re.sub(u"７", u"7", text)
            text = re.sub(u"８", u"8", text)
            text = re.sub(u"９", u"9", text)
            text = re.sub(r"查看原文>>", "", text)
            text = re.sub(r"显示图片", "", text)
            text = re.sub(r"您的浏览器不支持video标签。", "", text)
            text = re.sub(r"{.*}", "", text)
            #text = re.sub(r"【.+?】", "", text)
            text = re.sub(r"\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{4}-\d{1,2}-\d{1,2}\d{1,2}:\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时\d{1,2}分", "TIME", text)
            text = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日\d{1,2}时", "TIME", text)
            text = re.sub(r"\d{1,2}月\d{1,2}日\d{1,2}时\d{1,2}分", "TIME", text)
            text = re.sub(r"\d{1,2}月\d{1,2}日\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{4}-\d{1,2}-\d{1,2}\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{1,2}-\d{1,2}\d{1,2}:\d{1,2}", "TIME", text)
            text = re.sub(r"\d{1,2}月\d{1,2}日-\d{1,2}日", "TIME", text)
            text = re.sub(r"\d{4}年\d{1,2}月\d{1,2}日", "TIME", text)
            text = re.sub(r"\d{1,2}日\d{1,2}时\d{1,2}分", "TIME", text)
            text = re.sub(r"\d{1,2}月\d{1,2}日", "TIME", text)
            text = re.sub(r"\d{1,2}月\d{1,2}", "TIME", text)
            text = re.sub(r"\d{1,2}时\d{1,2}分", "TIME", text)
            text = re.sub(r"\d{1,2}点\d{1,2}分", "TIME", text)
            text = re.sub(r"\d{4}年\d{1,2}月", "TIME", text)
            text = re.sub(r"\d{1,2}:\d{1,2}分", "TIME", text)
            text = re.sub(r"\d{4}年", "TIME", text)
            #text = re.sub(r"\d{1,2}{月|日|时|点}", "TIME", text)
            text = re.sub(r"(\d+)(\.\d+)?", "NUM", text)
            f.write(text)
    f.close()

#clean_data(r"article")
#clean_data(r"summarization")


def data_segment(filename):
    n_file = "data" + "\\" + filename + "_segment.txt"
    f = open(n_file, 'w+', encoding="utf-8")
    for line in open("data"+"\\"+filename+"_cleaned.txt", 'r+', encoding="utf-8"):
        f.write(" ".join(jieba.cut(line)))
    f.close()


#data_segment(r"article")
#data_segment(r"summarization")


def extract_character_vocab(source_data, target_data):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    # 以行分割再以空格分割，统计分词后中文
    source_words = list(set([character for line in source_data.split('\n') for character in line.split()]))
    target_words = list(set([character for line in target_data.split('\n') for character in line.split()]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + source_words + target_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def loadDataset(source_file, target_file):
    # 读取源文件
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = f.read()
    # 读取目标文件
    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = f.read()
    # 构造映射表
    # source_int_to_segment, source_segment_to_int = extract_character_vocab(source_data)
    # target_int_to_segment, target_segment_to_int = extract_character_vocab(target_data)
    int_to_vocab, vocab_to_int = extract_character_vocab(source_data, target_data)
    # 对字分词进行转换
    # 对于source来说 不需要添加<EOS>标记
    source_int = [[vocab_to_int.get(letter, vocab_to_int['<UNK>'])
                   for letter in line.split()] for line in source_data.split('\n')]
    # 对于target来说 需要添加<EOS>标记
    target_int = [[vocab_to_int.get(letter, vocab_to_int['<UNK>'])
                   for letter in line.split()] + [vocab_to_int['<EOS>']] for line in target_data.split('\n')]
    return [int_to_vocab, vocab_to_int], \
           [source_int, target_int]


# 创建批次 每次使用getBatch方法后得到一个经过填充的批次即可
def get_batches(data, batch_size, source_pad_int, target_pad_int):
    sources, targets = data
    data_len = len(sources)
    for batch_i in range(0, data_len, batch_size):
        sources_batch = sources[batch_i:min(batch_i + batch_size, data_len)]
        targets_batch = targets[batch_i:min(batch_i + batch_size, data_len)]
        # 补全序列
        padded_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        padded_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每一条记录的长度
        source_sequence_length = []
        target_sequence_length = []
        for s in sources_batch:
            source_sequence_length.append(len(s))
        for t in targets_batch:
            target_sequence_length.append(len(t))

        yield padded_sources_batch, padded_targets_batch, source_sequence_length, target_sequence_length


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

