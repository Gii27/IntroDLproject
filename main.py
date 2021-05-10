from char import CharModel
from word import WordModel
from common import *
import matplotlib.pyplot as plt
import numpy as np
import sys


def main():
    # train_char, train_tag = read_char_tag('data/MSRA/BIO_train.txt')
    # test_char, test_tag = read_char_tag('data/MSRA/BIO_test.txt')
    train_char, train_tag = read_char_tag('data/train.txt')
    test_char, test_tag = read_char_tag('data/test.txt')
    fig, ax = plt.subplots()
    ind = np.arange(3)
    width = 0.3
    batch_size = 256
    
    mode = sys.argv[1]
    if len(sys.argv) == 2:
        dropout1 = 0
        dropout2 = 0
    elif len(sys.argv) == 4:
        dropout1 = float(sys.argv[2])
        dropout2 = float(sys.argv[3])

    if mode == "word":
        word_model = WordModel(train_char, train_tag, batch_size, dropout1, dropout2)

        test_input = ['今年', '第一季度', '，', '广州市', '私营', '科技咨询', '业已', '达', '五百三十', '四家', '，', '比', '去年同期', '增加一倍', '以上', '。']
        word_model.predict(test_input)

        word_tag = word_model.predict(test_char)
        word_score = score(word_tag, test_tag)
        word_bar = ax.bar(ind, word_score, width)
        precision, recall, f1 = word_score
        print(f"precision = {precision}, recall = {recall}, f1-score = {f1}")

    elif mode == "char":
        char_model = CharModel(train_char, train_tag, batch_size, dropout1, dropout2)
        char_tag = char_model.predict(test_char)
        char_score = score(char_tag, test_tag)
        char_bar = ax.bar(ind + width, char_score, width)
        precision, recall, f1 = char_score
        print(f"precision{precision}, recall{recall}, f1-score{f1}")

    # ax.set_ylabel('score')
    # ax.set_title('comparison')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(('precision', 'recall', 'F1'))
    # plt.ylim([0, 100])
    # ax.legend((word_bar[0], char_bar[0]), ('Word Based', 'Character Based'))
    # ax.autoscale_view()

    # plt.grid()
    # plt.show()



if __name__ == '__main__':
    main()
