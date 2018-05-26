#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/24 15:56
# @Author  : Zhangyuheng
# @File    : pre_data.py


# def format_data(filename):
#     start = False
#     text = []
#     meaning = []
#     num = 0
#     with open(filename, 'r') as file:
#         for line in file.readlines():
#             if line.strip() == '------------':
#                 start = True
#                 continue
#             if num == 0:
#                 num = 1
#             elif num == 1:
#                 num = 2
#             elif num == 2:
#                 num = 3
#             elif num == 3:
#                 text.append(line.strip())
#                 num = 4
#             elif num == 4:
#                 meaning.append(line.strip())
#                 num = 0
#             else:
#                 raise EOFError
#
#     return text, meaning
#
#
# text, meaning = format_data('data2/伊索寓言.txt')
# print(len(text))
# print(len(meaning))


# def check(filename):
#     with open(filename, 'r') as file:
#         for num, line in enumerate(file.readlines()):
#             if num % 6 == 0:
#                 if '------------' != line.strip():
#                     print(num)
#                     break
#
#
# check('data2/伊索寓言.txt')

'''
数据预处理
train_with_summ.txt  训练
evaluation_without_ground_truth.txt  评估
1. 根据数据集构建字典（50000条最常用分词 剩余的词典）
'''