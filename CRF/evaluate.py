#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：GraduationProject -> evaluate
@IDE    ：PyCharm
@Author ：Wang hai
@Date   ：2021/4/29 20:32
@Desc   ：
=================================================='''
from CRF.crf import CRFModel
from HMM.utils import save_model
from HMM.evaluating import Metrics

def crf_train_eval(train_data, test_data, remove_O=False):

    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "../FinalModel/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists