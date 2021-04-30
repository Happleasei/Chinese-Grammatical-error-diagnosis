#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraduationProject -> main_bilstm_crf
@IDE    ：PyCharm
@Author ：Wang hai
@Date   ：2021/4/29 20:51
@Desc   ：
=================================================='''
from HMM.data import build_corpus
from BILSTM.evaluate import bilstm_train_and_eval
from HMM.utils import extend_maps, prepocess_data_for_lstmcrf


print("读取数据...")
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")
dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

print("正在训练评估Bi-LSTM+CRF模型...")
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
# 还需要额外的一些数据处理
train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
    train_word_lists, train_tag_lists
)
dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
    dev_word_lists, dev_tag_lists
)
test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
    test_word_lists, test_tag_lists, test=True
)
lstmcrf_pred = bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    (test_word_lists, test_tag_lists),
    crf_word2id, crf_tag2id
)