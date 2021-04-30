#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraduationProject -> MAIN
@IDE    ：PyCharm
@Author ：Wang hai
@Date   ：2021/4/29 20:29
@Desc   ：
=================================================='''
from CRF.evaluate import crf_train_eval
from HMM.data import build_corpus

# 读取数据
print("读取数据...")
train_word_lists, train_tag_lists, word2id, tag2id = \
    build_corpus("train")
dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

# # 训练评估CRF模型
print("正在训练评估CRF模型...")
crf_pred = crf_train_eval(
    (train_word_lists, train_tag_lists),
    (test_word_lists, test_tag_lists))