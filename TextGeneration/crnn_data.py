import codecs
import numpy as np
import pickle

class CRNNDdata:
    def __init__(self,file_path,batch_line,time_step):
         self.file_path = file_path
         self.batch_line = batch_line
         self.time_step = time_step
         self.batch_size = batch_line * time_step
         self.text = self.get_text(file_path)
         self.char_number_dict, self.number_char_dict = self.char_or_number_dict()
         self.text_number_list = self.text_to_number()
         return

    def get_text(self,file_path):
        with codecs.open(self.file_path, mode='r', encoding='utf-8') as f:
            return f.read()

    def char_or_number_dict(self):
        chars = set(self.text)
        char_nums_dict = {}
        for char in chars:
            char_nums_dict[char] = 0
        for char in self.text:
            char_nums_dict[char] += 1
        char_nums_lst = sorted(char_nums_dict.items(),key=lambda x:x[1],reverse=True)
        char_lst = [item[0] for item in char_nums_lst]
        char_number_dict = {char:number for number,char in enumerate(char_lst)}
        number_char_dict = dict(enumerate(char_lst))
        return char_number_dict,number_char_dict

    def text_to_number(self):
        number = []
        for i in self.text:
            number.append(self.char_number_dict[i])
        return number

    def get_batch_data(self):
        # 计算batch数量
        batch_number = int(len(self.text_number_list) / self.batch_size)
        self.text_number_list = self.text_number_list[:self.batch_size * batch_number]
        text_number_np = np.array(self.text_number_list).reshape((self.batch_line,-1))
        while True:
            np.random.shuffle(text_number_np)
            for i in range(0,batch_number):
                X = text_number_np[:,self.time_step * i:self.time_step * i + self.time_step]
                Y = np.zeros_like(X)
                Y[:,:-1] = X[:,1:]
                Y[:,-1] = X[:,0]
                yield X,Y

    def save_dict(self,dict_directory_path):
        with open(dict_directory_path + "crnn_char_number_dict.pkl","wb") as f:
            pickle.dump(self.char_number_dict,f)
        with open(dict_directory_path + "crnn_number_char_dict.pkl", "wb") as f:
            pickle.dump(self.number_char_dict,f)


def main():
    crnn_data = CRNNDdata("./data/shakespeare.txt",50,32)
    for X,Y in crnn_data.get_batch_data():
        print(type(X))
        print(Y.shape)

if __name__ == "__main__":
    main()