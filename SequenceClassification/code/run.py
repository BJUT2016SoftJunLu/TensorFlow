import numpy as np
from code.sc_model import *
from code.data_generation import *

def train():
    sc_model = SCModel(hidden_layer=64,
                       learn_rate=0.01,
                       batch_rows=1000,
                       max_length=20,
                       class_nums=2,
                       max_iteration=10000,
                       save_model_path="../model/sc_model")
    loss, optimizer, _ = sc_model.build()
    sc_model.train(loss,optimizer)
    return

def predict():
    tsd = ToySequenceData(n_samples=500, max_seq_len=20)
    test_data = np.array(tsd.data)
    test_seqlen = np.array(tsd.seqlen)

    sc_model = SCModel(hidden_layer=64,
                       learn_rate=0.01,
                       batch_rows=500,
                       max_length=20,
                       class_nums=2,
                       load_model_directory="../model/")
    _, _, prob = sc_model.build()
    session=  sc_model.load_model()
    prob_result = sc_model.predict(session,prob,test_data,test_seqlen)[0]
    class_result = np.zeros_like(prob_result)
    for i in range(prob_result.shape[0]):
        index = np.argmax(prob_result[i])
        class_result[i][index] = 1
    print(class_result)
    return

def main():

    return

if __name__ == '__main__':
     main()

