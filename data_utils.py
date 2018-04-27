#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 23:21
# @Author  : Zhangyuheng
# @File    : data_utils.py

import numpy as np
import time
import tensorflow as tf
import os


class Batch:
    #batch类，里面包含了encoder输入，decoder输入以及他们的长度
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []

def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    # 以行分割再以空格分割，统计分词后中文
    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def loadDataset(source_file, target_file):
    # 读取源文件
    with open(source_file, 'rb', encoding='utf-8') as f:
        source_data = f.read()
    # 读取目标文件
    with open(target_file, 'rb', encoding='utf-8') as f:
        target_data = f.read()
    # 构造映射表
    source_int_to_segment, source_segment_to_int = extract_character_vocab(source_data)
    target_int_to_segment, target_segment_to_int = extract_character_vocab(target_data)

    # 对字分词进行转换
    # 对于source来说 不需要添加<EOS>标记
    source_int = [[source_segment_to_int.get(letter, source_segment_to_int['<UNK>'])
                   for letter in line] for line in source_data.split('\n')]
    # 对于target来说 需要添加<EOS>标记
    target_int = [[target_segment_to_int.get(letter, target_segment_to_int['<UNK>'])
                   for letter in line] + [target_segment_to_int['<EOS>']] for line in target_data.split('\n')]
    return [source_int_to_segment, source_segment_to_int], \
           [target_int_to_segment, target_segment_to_int], \
           [source_int, target_int]


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence_batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        padded_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        padded_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield padded_targets_batch, padded_sources_batch, targets_lengths, source_lengths

