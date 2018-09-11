# -*- coding: utf-8 -*-

import re

def format_print_han_list(l1, l2, space_num=12):
    re_han_cut_all = re.compile("[\u4E00-\u9FD5]", re.U)

    space = space_num
    for idx, (x, y) in enumerate(zip(l1, l2)) :
        space1 = space
        space2 = space

        for e in x:
            if re_han_cut_all.match(e):
                space1 -= 2
            else :
                space1 -= 1
        for e in y:
            if re_han_cut_all.match(e):
                space2 -= 2
            else :
                space2 -= 1

        l1[idx] = ' '*space1 + l1[idx]
        l2[idx] = ' '*space2 + l2[idx]

    return l1, l2
    # print("{0:^6}\t{1:{3}^10}\t{2:^6}".format(l1[0],ul[1],ul[2],chr(12288)))


if __name__ == "__main__" :
    l1 = ['kaix', '开心的锣鼓', '3', '额']
    l2 = ['34r32', '2348是', 'weir', '<UNK>']

    o1, o2 = format_print_han_list(l1, l2, space_num=12)

    with open('temp.txt', 'w', encoding='utf8') as writer:
        writer.write(' '.join(o1) + '\n')
        writer.write(' '.join(o2) + '\n')