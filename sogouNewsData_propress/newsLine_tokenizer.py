# -*- coding: utf-8 -*-

"""
@author: weijiawei
@date: 2018-08-10
PTB的数据格式是：将每个句子用\n来隔开。
本文件将搜狗新闻语料 处理成与PTB一致的格式。
"""

import os, sys
import jieba
import re

from my_tools.my_files import MyFiles


data_input_path = os.path.join('..','..', 'sogou_news_line')
data_output_path = os.path.join('..','input', 'sogouNewsdata')

end_token = ['。', '！', '？', '.', '!', '?', '\u3000']    # 句子结束符号
filter_token = ['"', "“", "”", '\ue40c']    # 需要过滤掉的符号
forbid_token = ['责任编辑', '跳转']   # 禁止出现的符号。一旦出现在某个句子里，该句子直接丢弃

digit_re = re.compile(r"^(-?\d+)(\.\d*)?%?$")   # 匹配负号、数字、小数、百分号数
TAG_NUMBER = 'TAG_NUMBER'   # 所有数字被替换为 字符串TAG_NUMBER
MIN_WORDS_NUM = 5


if __name__ == '__main__':
    ret = []

    input_files = MyFiles(data_input_path)
    for file_idx,file_name in enumerate(input_files.base_file_name()):
        with open(os.path.join(data_input_path, file_name), 'r', encoding='utf8') as fin, \
                open(os.path.join(data_output_path, file_name), 'w', encoding='utf8') as fout:
            for line in fin.readlines():
                # 拆分出句子
                sentences = re.split(r'['+''.join(end_token)+']', line)
                for sentence in sentences:
                    # 分词
                    words = list(jieba.cut(sentence))
                    # 含有禁止词的句子 直接丢弃；将数字替换成TAG_NUMBER
                    forbid_flag = False
                    for idx, word in enumerate(words):
                        if word in forbid_token:
                            forbid_flag = True
                            break
                        if re.match(digit_re, word):
                            words.pop(idx)
                            words.insert(idx, TAG_NUMBER)
                    if forbid_flag : continue

                    # 丢弃空白词 和 过滤词
                    words = [x for x in words if len(x.strip()) and (x not in filter_token)]
                    # 丢弃太短的句子
                    if len(words) < MIN_WORDS_NUM: continue
                    # 将句子写入文件
                    fout.write(' '.join(words) + '\n')

        if file_idx %10 ==0 :
            print('propress %d files..' % file_idx)
        # for debug
        break
