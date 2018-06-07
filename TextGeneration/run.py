import crnn_data
import crnn_model
import pickle
import tensorflow as tf

def train():
    data = crnn_data.CRNNDdata("./data/shakespeare.txt", batch_line=100, time_step=100)
    model = crnn_model.CRNNModel(lstm_size=128,
                       lstm_layer=2,
                       learning_rate=0.01,
                       keep_prop=0.5,
                       init_state_size=100,
                       batch_line=100,
                       time_step=100,
                       class_nums=len(data.char_number_dict.keys()) + 1,
                       data_generator=data.get_batch_data(),
                       save_every_batch=100,
                       log_every_batch=10,
                       save_model_path="./model/TextGeneration")
    model.build()
    model.train()
    data.save_dict("./chars/")
    return

def predict():
    with open("./chars/crnn_char_number_dict.pkl","rb") as f:
        char_number_dict = pickle.load(f)
    with open("./chars/crnn_number_char_dict.pkl", "rb") as f:
        number_char_dict = pickle.load(f)

    print(number_char_dict)

    model = crnn_model.CRNNModel(lstm_size=128,
                                 lstm_layer=2,
                                 learning_rate=0.01,
                                 keep_prop=1,
                                 init_state_size=1,
                                 batch_line=1,
                                 time_step=1,
                                 class_nums=len(char_number_dict.keys()) + 1,
                                 load_model_directory="./model")

    model.build()
    model.load_model()
    return model.predict(init_char_number=8,
                         predict_length=100,
                         number_char_dict=number_char_dict)


def main():
    print(predict())
    return

if __name__ == '__main__':
    main()