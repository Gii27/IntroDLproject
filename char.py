import numpy as np
from common import *
from model import BiLSTM_CRF
from gpu_model import BiLSTM_CRF_gpu

class CharModel:
    def __init__(self,train_char:list,train_tag:list, gpu, epoch=100, batch_size=1, lr=0.01, dropout1=0, dropout2=0):
        hidden_dim=100
        if not gpu:
            self.model=BiLSTM_CRF('data/char_vec.txt',hidden_dim=hidden_dim, sent_list=train_char, tag_list=train_tag, mode="char",
                        epoch=epoch, learning_rate=lr, dropout1=dropout1, dropout2=dropout2)
        else:
            self.model = BiLSTM_CRF_gpu('data/char_vec.txt', hidden_dim=hidden_dim, sent_list=train_char,
                                    tag_list=train_tag, mode="char",
                                    epoch=epoch, batch_size=batch_size, learning_rate=lr, dropout1=dropout1, dropout2=dropout2)

    def predict(self, char_list: list) -> list:
        return self.model.predict(char_list)

def main():
    train_char, train_tag = read_char_tag('data/train.txt')
    test_char, test_tag = read_char_tag('data/test.txt')
    char_model = CharModel(train_char, train_tag)
    char_tag = char_model.predict(test_char)
    char_p,char_r,char_f = score(char_tag, test_tag)
    write_char_tag(test_char,char_tag,'data/char_output.txt')
    print('precision={}, recall={}, F1={}'.format(char_p,char_r,char_f))

if __name__ == '__main__':
    main()