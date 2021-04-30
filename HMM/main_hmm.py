#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraduationProject -> MAIN
@IDE    ：PyCharm
@Author ：Wang hai
@Date   ：2021/4/29 19:56
@Desc   ：
=================================================='''
from HMM.data import build_corpus
from HMM.evaluate import hmm_train_eval

def main():
    """训练模型，评估结果"""

    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    print(tag2id)
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )

if __name__ == "__main__":
    main()