# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 64          # 64
    # 学习速率
    lr = 0.0151                #0.01
    epoches = 30               # 30可以出结果
    print_step = 5


class LSTMConfig(object):
    emb_size = 100  # 词向量的维数
    hidden_size = 100  # lstm隐向量的维数
