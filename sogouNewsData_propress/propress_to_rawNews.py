# coding: utf8

import os
from bs4 import BeautifulSoup


fin_floder = os.path.join('..','..','..', 'SogouCS.reduced')
data_output_path = os.path.join('..','..','..', 'sogou_news')
MIN_LEN = 10


if __name__=='__main__':

    for idx,file_name in enumerate(os.listdir(fin_floder)):
        with open(os.path.join(fin_floder, file_name), 'r', encoding='gb18030', errors='ignore') as fin, \
                open(os.path.join(data_output_path, file_name), 'w', encoding='utf8') as fout:
            soup = BeautifulSoup(fin.read(), 'html.parser')
            # print(soup.contents)

            def check(elem) :
                """检查elem是否符合规范。不符合规范返回False"""
                if not(elem and len(elem.text.strip())>MIN_LEN) :
                    return False
                else :
                    return True


            def DBC2SBC(ustring):
                # 全角转半角
                rstring = ""
                for uchar in ustring:
                    inside_code = ord(uchar)
                    if inside_code == 0x3000:
                        inside_code = 0x0020
                    else:
                        inside_code -= 0xfee0
                    if not (0x0021 <= inside_code and inside_code <= 0x7e):
                        rstring += uchar
                        continue
                    rstring += chr(inside_code)
                return rstring

            for doc in soup.findAll('doc'):
                url = doc.find('url')
                contenttitle = doc.find('contenttitle')
                content = doc.find('content')

                if not (check(url) and check(contenttitle) and check(content)) :
                    continue
                if contenttitle.text=='本公司电子公告服务严格遵守':
                    continue

                fout.writelines(url.text+'`1`2'+DBC2SBC(contenttitle.text)+'`1`2'+DBC2SBC(content.text)+'\n')

        if idx %10 ==0 :
            print('propress %d files..' % idx)
        # break