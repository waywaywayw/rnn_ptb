import collections
import os
import numpy as np
import sys
sys.path.append("..")
from data import *

data_input_path = os.path.join('..', '..', '..', 'sogou_news_line_tokenized')
data_output_path = os.path.join('my_vocab.txt')
MIN_WORD_NUM = 10

def _read_words(data_input_path):
    for idx, file_name in enumerate(os.listdir(data_input_path)):
        with open(os.path.join(data_input_path, file_name), "r", encoding='utf8') as fin:
            for line in fin:
                words = line.split()
                for word in words:
                    yield word

        if idx %20 ==0:
            print('propress %d files..' % idx)


# 输入文件路径，返回word_to_idx
def create_dictionary(data_input_path, data_output_path):
    counter = collections.Counter()

    words = _read_words(data_input_path)
    for word in words:
        counter[word] += 1
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    with open(data_output_path, 'w', encoding='utf8') as fout:
        for words, freq in count_pairs:
            if freq > MIN_WORD_NUM:
                fout.write('{} {}\n'.format(words, freq))

        # 添加UNK、 PAD
        fout.write('{} {}\n'.format(UNKNOWN_TOKEN, 233))
        fout.write('{} {}\n'.format(PAD_TOKEN, 5))
    # words, freq = list(zip(*count_pairs))
    # word_to_id = dict(zip(words, range(len(words))))

    # return word_to_id


if __name__ == "__main__":
    create_dictionary(data_input_path, data_output_path)
