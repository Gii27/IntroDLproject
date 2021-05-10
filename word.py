import numpy as np
from common import *
from model import BiLSTM_CRF
from gpu_model import BiLSTM_CRF_gpu
import jieba

class WordModel:
    def __init__(self, train_char:list,train_tag:list, gpu, epoch=100, batch_size=1, lr=0.01, dropout1=0, dropout2=0):
        hidden_dim=100
        train_word,train_chunk=char2word(train_char,train_tag)
        if not gpu:
            self.model=BiLSTM_CRF('data/word_vec.txt', hidden_dim=50, sent_list=train_word, tag_list=train_chunk, mode="word",
                        epoch=epoch, learning_rate=lr, dropout1=dropout1, dropout2=dropout2)
        else:
            self.model=BiLSTM_CRF_gpu('data/word_vec.txt', hidden_dim=50, sent_list=train_word, tag_list=train_chunk, mode="word",
                        epoch=epoch, batch_size=batch_size, learning_rate=lr, dropout1=dropout1, dropout2=dropout2).cuda()

    def predict(self, char_list: list) -> list:
        word_list=word_seg(char_list)
        chunk_list = self.model.predict(word_list)
        return chunk2tag(word_list,chunk_list)


def main():
    train_char, train_tag = read_char_tag('data/train.txt')
    test_char, test_tag = read_char_tag('data/test.txt')
    word_model = WordModel(train_char, train_tag)
    char_tag = word_model.predict(test_char)
    word_p,word_r,word_f = score(char_tag, test_tag)
    write_char_tag(test_char,char_tag,'data/word_output.txt')
    print('precision={}, recall={}, F1={}'.format(word_p,word_r,word_f))

    # char_list = [['武', '汉', '加', '油'], ['乌', '鸦', '坐', '飞', '机']]
    # tag_list = [['B-LOC', 'E-LOC', 'O', 'O'], ['O', 'O', 'O', 'O', 'O']]
    # word_list, chunk_list = [['武汉', '加油']], [['LOC', 'O']]
    # newchar, newtag = char2word(char_list, tag_list)
    # print(newchar)
    # print(newtag)
    # ans = chunk2tag(newchar, newtag)
    # print(ans)

def char2word(sent_list:list,tag_list:list)->(list,list):
    out_tag = []

    word_lists = word_seg(sent_list)
    for sen, tags in zip(word_lists, tag_list):
        length = 0
        new_tag_list = []
        for word in sen:
            tag = tags[length]
            if tag != "O":
                _, tag = tag.split("-")
            new_tag_list.append(tag)
            length += len(word)
        out_tag.append(new_tag_list)

    return word_lists, out_tag
    # return [['陈元','呼吁','加强','合作'],['武汉','加油']],[['PER','O','O','O'],['LOC','O']]

def word_seg(char_list:list)->list:
    # char_list=[['武','汉','加','油']]
    ans = []
    for sen in char_list:
        sentence = ""
        for char in sen:
            sentence += char
        words = jieba.cut(sentence)
        ans.append(list(words))
    # print(ans)
    return ans
    # return [['武汉','加油']]
def chunk2tag(word_list:list,chunk_list:list)->list:
    # word_list,chunk_list=[['武汉','加油']],[['LOC','O']]
    ans = []
    for sen, chunk in zip(word_list, chunk_list):
        tag_list = []
        for word, tag in zip(sen, chunk):
            if len(word) == 1:
                if tag == "O":
                    tag_list.append(tag)
                else:
                    tag_list.append("S-" + tag)
            else:
                for i in range(len(word)):
                    if tag == "O":
                        tag_list.append(tag)
                    else:
                        if i == 0:
                            tag_list.append("B-" + tag)
                        elif i == len(word) - 1:
                            tag_list.append("E-" + tag)
                        else:
                            tag_list.append("M-" + tag)
        ans.append(tag_list)

    return ans
    # return [['B-LOC','E-LOC','O','O']]

if __name__ == '__main__':
    main()
