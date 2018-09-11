import os
import sys
import logging


class MyFiles(object):
    def __init__(self, fin_path):
        """
        如果是文件夹，不会深层遍历文件夹
        fin_path: 可以是文件夹名，也可以是单个文件名
        """
        if not os.path.isfile(fin_path) and not os.path.exists(fin_path):
            print("fin_path is not a file and not a folder path.")
            raise TypeError

        self.fin_folder = ""
        self.fin_files = []

        if os.path.isfile(fin_path):
            splited = os.path.split(fin_path)
            self.fin_folder = splited[0]
            self.fin_files.append(splited[1])
        else:
            self.fin_folder = fin_path
            self.fin_files = os.listdir(self.fin_folder)
            # 过滤掉 fin_folder 中的文件夹，只留下文件
            self.fin_files = filter( \
                lambda x: os.path.isfile(os.path.join(self.fin_folder, x)), \
                self.fin_files)
            # logging.info(self.fin_files)

    def __iter__(self):
        """
        返回带完整路径的文件名
        :return: generator of file list
        """
        for file in self.fin_files:
            yield os.path.join(self.fin_folder, file)

    def base_file_name(self):
        """
        返回不带路径的文件名
        :return:
        """
        for file in self.fin_files:
            yield file


class MyVocab(object):
    """Vocabulary class for mapping words and ids."""

    def __init__(self, vocab_file, max_size, UNKNOWN_TOKEN='<UNK>'):
        """
        从一个字典文件中初始化字典类。
        字典文件的指定格式为：
        单词 频数
        单词 频数
        ...
        :param vocab_file:
        :param max_size:
        :param UNKNOWN_TOKEN: 未知单词对应的字符串
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self.UNKNOWN_TOKEN = UNKNOWN_TOKEN

        with open(vocab_file, 'r', encoding='utf8') as vocab_f:
            for line in vocab_f:
                # pieces[0]是单词, pieces[1]是词频
                pieces = line.split()
                # 两个异常处理
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                # 存入word_to_id
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                self._count += 1
                if self._count >= max_size-1:
                    print('words size = %d.' % max_size)
                    break

            # 添加未知标记
            self._word_to_id[self.UNKNOWN_TOKEN] = self._count
            self._id_to_word[self._count] = self.UNKNOWN_TOKEN
            self._count += 1

    def CheckVocab(self, word):
        """返回对应word的index.
            如果找不到，返回None.
        """
        if word not in self._word_to_id:
            return None
        return self._word_to_id[word]

    def WordToId(self, word):
        """返回对应word的index.
            如果找不到，返回 UNKNOWN_TOKEN 对应的index.
        """
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        """返回字典大小"""
        return self._count

    def word_to_id(self):
        return self._word_to_id

    def id_to_word(self):
        return self._id_to_word


"""
准备写：
自动按比例分成训练集，验证集，测试集
split
"""
