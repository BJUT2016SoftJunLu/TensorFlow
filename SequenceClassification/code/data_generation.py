import random
import  numpy as np


class ToySequenceData(object):
    """
    生成序列数据
    - 类别 0: 线性序列 (如 [0, 1, 2, 3,...])
    - 类别 1: 完全随机的序列 (i.e. [1, 3, 10, 7,...])
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # 序列的长度是随机的，在min_seq_len和max_seq_len之间。
            len = random.randint(min_seq_len, max_seq_len)
            # self.seqlen用于存储所有的序列。
            self.seqlen.append(len)
            # 以50%的概率，随机添加一个线性或随机的训练
            if random.random() < .5:
                # 生成一个线性序列
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # 长度不足max_seq_len的需要补0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                # 线性序列的label是[1, 0]（因为我们一共只有两类）
                self.labels.append([1., 0.])
            else:
                # 生成一个随机序列
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # 长度不足max_seq_len的需要补0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """
        生成batch_size的样本。
        如果使用完了所有样本，会重新从头开始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

def main():
    tsd = ToySequenceData()
    batch_data, batch_labels, batch_seqlen = tsd.next(32)

    print(np.array(batch_data))
    print(np.array(batch_labels).shape)
    # print(np.array(batch_seqlen).shape)
    #
    # test_data = tsd.data
    # test_seqlen = tsd.seqlen
    # print(np.array(test_data))
    # print(np.array(test_seqlen))

    return

if __name__ == '__main__':
    main()