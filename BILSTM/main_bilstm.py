#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraduationProject -> main_bilstm
@IDE    ：PyCharm
@Author ：Wang hai
@Date   ：2021/4/29 20:41
@Desc   ：
=================================================='''
from HMM.data import build_corpus
from BILSTM.evaluate import bilstm_train_and_eval
from HMM.utils import extend_maps
# 读取数据
print("读取数据...")
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")
dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

# 训练评估BI-LSTM模型
print("正在训练评估双向LSTM模型...")
# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
lstm_pred = bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    (test_word_lists, test_tag_lists),
    bilstm_word2id, bilstm_tag2id,
    crf=False
)