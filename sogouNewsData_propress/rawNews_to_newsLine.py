# -*- coding: utf-8 -*-

import os

data_input_path = os.path.join('..','..', 'sogou_news')
data_output_path = os.path.join('..','..', 'sogou_news_line')


if __name__=='__main__':
    for idx,file_name in enumerate(os.listdir(data_input_path)):
        with open(os.path.join(data_input_path, file_name), 'r', encoding='utf8') as fin, \
                open(os.path.join(data_output_path, file_name), 'w', encoding='utf8') as fout :
            for line in fin.readlines():
                elems = line.split('`1`2')
                fout.write(elems[2]+elems[1]+'\n'+'\n')

        if idx %10 ==0 :
            print('propress %d files..' % idx)
        # break

    fout.close()