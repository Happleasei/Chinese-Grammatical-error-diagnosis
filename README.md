# Chinese-Grammatical-error-diagnosis
# 中文语法纠错研究 基于序列标注的方法

数据集采用的CGED官方提供
转换为序列标注的形式 具体可以看Data中的数据

采用的几个模型如下：
HMM\CRF\BILSTM\BERT-CRF\BILSTM-CRF\BERTBILSTMCRF\XLNETBILSTMCRF

以上模型均修改调优自命名实体识别的代码，并非原创，侵权告诉我删除

前五个模型 环境为python3.7 基于pytorch-gpu 具体怎么合适 根据你的显卡自己配置cuda等等
最后两个模型 环境为pthon3.6 基于tensorflow-gpu

所有程序都在PyCharm上执行，效果一般，xlnet bilstm crf/HMM效果很差，如果你找出原因，麻烦告诉我

部分程序需要，加入parameter，内容在主函数首行

# 有任何问题欢迎私聊

# 此外本人研二在读，如果有nlp实习开发的工作欢迎邀请我笔试面试 感谢
