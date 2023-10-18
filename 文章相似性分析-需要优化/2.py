#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :2.py
# @Time      :2023/9/18 13:55
# @Author    :LongFei Shan
import pandas as pd
import pdfplumber
import docx
import fitz
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from gensim import matutils
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import tqdm
import time
from functools import wraps
import re
import colorama
import jieba
from multiprocessing import Pool, cpu_count
import gensim


# 加载默认的jieba分词词典
jieba.setLogLevel(20)  # 隐藏jieba的日志输出

class CompareFile:
    def __init__(self, name_distance="cosine", name_encode="tfidf", threshold_score=0.5, threshold_text_length=5, pages=None, laparams=None, password=None,
                 strict_metadata=False, repair=False, format=1, jpg_quality=94):
        """

        :param name_distance:  计算方式, 有cosine, euclidean
        :param name_encode:  编码方式, 有tfidf, hash, count, word2vec, bert
        若选择word2vec, 则需要下载word2vec-google-news-300模型， 若出现information错误，请参考：https://blog.csdn.net/weixin_43213884/article/details/116270090
        若在线方式连接不上，采用离线方式计算，参考： https://blog.csdn.net/m0_54882482/article/details/129281840
        :param threshold_score:  对于大于threshold_score的指标才算重复
        :param threshold_text_length: 小于threshold_text_length的文本不进行查重
        :param pages:  读取pdf文件的页码
        :param laparams:  pdfplumber的参数
        :param password:  pdf文件的密码
        :param strict_metadata:
        :param repair:
        :param format: 图片格式 , "png": 1, "pnm": 2, "pgm": 2, "ppm": 2, "pbm": 2,
        :param jpg_quality: 图片质量 0-100
        """
        self.name_distance = name_distance
        self.name_encode = name_encode
        self.threshold_score = threshold_score
        self.threshold_text_length = threshold_text_length
        self.pages = pages
        self.laparams = laparams
        self.password = password
        self.strict_metadata = strict_metadata
        self.repair = repair
        self.format = format
        self.jpg_quality = jpg_quality
        # self.word2vec = KeyedVectors.load_word2vec_format(
        #     datapath(r"C:\Users\31843\gensim-data\word2vec-google-news-300\GoogleNews-vectors-negative300.bin"),
        #     binary=True)
        self.word2vec = gensim.models.Word2Vec

    def __read_pdf(self, filename, pages=None, laparams=None, password=None, strict_metadata=False, repair=False):
        pdf = pdfplumber.open(filename, pages=pages, laparams=laparams, password=password,
                              strict_metadata=strict_metadata, repair=repair)
        pdf_image = fitz.open(filename)
        # 提取pdf文件中的图片与文字
        pages_text = []
        for page in pdf.pages:
            # 判断是否有图片
            try:
                image_list = pdf_image.get_page_images(page.page_number)
                if image_list:
                    self.__get_pdf_image(pdf_image, image_list, page.page_number, format=self.format, jpg_quality=self.jpg_quality)
            except:
                pass
            # 获取当前页的全部文本信息，包括表格中的文字
            text = page.extract_text()
            if text:
                # 分割文本
                text = self.__split_text(text)
                for t in text:
                    if len(t) > self.threshold_text_length:
                        pages_text.append([t, page.page_number])
        # 关闭pdf文件
        pdf_image.close()
        pdf.close()
        #输出绿色
        print(colorama.Fore.GREEN + f"文件{filename}读取完成")
        return pages_text

    def fit_transform(self, filename1, filename2):
        """
        计算两个文件的相似度
        :param filename1:
        :param filename2:
        :param name_distance: 计算方式, 有cosine, e
        :param name_encode: 编码方式, 有tfidf, hash, count, word2vec, bert
        :return:
        """
        pool = Pool(5)
        # 读取文件
        text1 = self.__read_pdf(filename=filename1, pages=self.pages, laparams=self.laparams, password=self.password,
                                strict_metadata=self.strict_metadata, repair=self.repair)
        text2 = self.__read_pdf(filename=filename2, pages=self.pages, laparams=self.laparams, password=self.password,
                                strict_metadata=self.strict_metadata, repair=self.repair)
        # 循环两两计算相似度
        result = []
        for i in range(len(text1)):
            for j in tqdm.tqdm(range(len(text2)), leave=False, desc=f"Epoch={i}/{len(text1)}-计算相似度", ncols=100, unit="个", colour="green"):
                try:
                    similarity = self.calculate_similarity_text(text1[i][0], text2[j][0], name_distance=self.name_distance, name_encode=self.name_encode)
                    result.append([text1[i][0], text2[j][0], text1[i][1], text2[j][1], similarity])  # 保存相似度, 文本1， 文本2，相似度， 页码1， 页码2
                except Exception as e:
                    print(colorama.Fore.RED + f"error: {e}")
        # 清洗数据
        result = self.__clean_with_similarity(result, self.threshold_score)
        # 保存结果
        self.__save_csv(result, "文章查重结果.csv")
        # 清空图片文件夹
        self.__clear_images()
        return result

    def __clean_with_similarity(self, result, threshold_score):
        """
        根据相似度清洗数据
        :param result:
        :param threshold_score:
        :return:
        """
        # 根据相似度清洗数据
        result = [r for r in result if r[-1] > threshold_score]
        return result

    def __split_text(self, text):
        # 采用常见的标点符号分割文本，、。！？；：（）.!?;:"'()[],若文中包含上述符号，则进行分割
        text =re.split(r'[,.。；;?！!:：（）()]', text)
        # 去除空格
        text = [t.replace(" ","") for t in text]
        # 去除非中文字符，数字，英文
        text = [re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]', '', t) for t in text]
        # 去除空字符串
        text = [t for t in text if t]
        return text

    def __save_csv(self, result, filename):
        """
        保存结果到excel
        :param result:
        :param filename:
        :return:
        """
        # 构造格式，编号，文本1，文本2，文本1-页码1，文本2-页码2，相似度， 保存到excel中
        pd.DataFrame(result, columns=["文本1", "文本2", "相似度", "文本1-页码", "文本2-页码"]).to_csv(filename, encoding="utf-8")

    def text_encode(self, text1, text2, name="tfidf"):
        """
        文本编码
        :param text1:
        :param text2:
        :return:
        """
        # 词频统计
        vec_fun = None
        text = [text1, text2]
        text_vec = None
        try:
            if name == "tfidf":
                vec_fun = TfidfVectorizer()
                # 计算个词语出现的次数
                vec_fun.fit(text)
                text_vec = vec_fun.transform(text).toarray()
            elif name == "hash":
                vec_fun = HashingVectorizer()
                # 计算个词语出现的次数
                vec_fun.fit(text)
                text_vec = vec_fun.transform(text).toarray()
            elif name == "count":
                vec_fun = CountVectorizer()
                # 计算个词语出现的次数
                vec_fun.fit(text)
                text_vec = vec_fun.transform(text).toarray()
            elif name == "word2vec":
                # 分词
                text = [self.__tokenize(t) for t in text]
                # 计算句向量
                text_vec1 = self.__sentence_to_vector(self.word2vec, text[0])
                text_vec2 = self.__sentence_to_vector(self.word2vec, text[1])
                if text_vec1 is None or text_vec2 is None:
                    return None
                text_vec = [text_vec1, text_vec2]
            elif name == "bert":
                model = SentenceTransformer('all-mpnet-base-v2')
                text_vec = model.encode(text)
            else:
                raise Exception("无效的编码方式")
        except Exception as e:
            print(colorama.Fore.RED + f"{text}, error: {e}")

        return text_vec

    def __tokenize(self, sentence):
        words = jieba.cut(sentence)
        return list(words)

    def __sentence_to_vector(self, word2vec_model, sentences):
        words = [sentence.split()[0] for sentence in sentences]
        word_vectors = []
        for word in words:
            if word in word2vec_model.key_to_index:
                word_vector = word2vec_model.get_vector(word)
                word_vectors.append(word_vector)
        if len(word_vectors) == 0:
            return None
        sentence_vector = np.mean(word_vectors, axis=0)
        return sentence_vector

    def calculate_distance(self, vec1, vec2, name="cosine"):
        """
        计算两段文字的相似度
        :param text1:
        :param text2:
        :param name: 计算方式, 有cosine, e
        :return:
        """
        # 计算余弦相似度
        similarity = None
        try:
            # 转换float
            vec1 = np.array(vec1).astype(np.float)
            vec2 = np.array(vec2).astype(np.float)
            # 查看两个向量是否为0向量
            if np.all(vec1 == 0) or np.all(vec2 == 0):
                return 0
            # 计算相似度
            if name == "cosine":
                similarity = cosine_similarity([vec1], [vec2])[0][0]
            elif name == "euclidean":
                similarity = distance.euclidean(vec1, vec2)
            else:
                raise Exception("无效的计算方式")
        except Exception as e:
            print(colorama.Fore.RED + f"vec1={vec1}, vec1_type={type(vec1)}, vec2={vec2}, vec2_type={type(vec2)}, error: {e}")
        return similarity

    def calculate_similarity_text(self, text1, text2, name_distance="cosine", name_encode="tfidf"):
        """
        计算两段文字的相似度
        :param text1:
        :param text2:
        :param name_distance: 计算方式, 有cosine, e
        :param name_encode: 编码方式, 有tfidf, hash, count, word2vec, bert
        :return:
        """
        similarity = None
        try:
            # 文本编码
            text_vec = self.text_encode(text1, text2, name=name_encode)
            if text_vec is None:
                return 0
            # 计算相似度
            similarity = self.calculate_distance(text_vec[0], text_vec[1], name=name_distance)
        except Exception as e:
            print(colorama.Fore.RED + f"text1={text1}, text2={text2}, error: {e}")
        return similarity

    def __get_pdf_image(self, pdf_image, image_list, index, format=1, jpg_quality=94):
        """
        获取pdf文件中的图片
        :param filename:  pdf文件名
        :param index:  第几页
        :param format:  图片格式 , "png": 1, "pnm": 2, "pgm": 2, "ppm": 2, "pbm": 2,
                         "pam": 3, "psd": 5, "ps": 6, "jpg": 7, "jpeg": 7
        :param jpg_quality:  图片质量 0-100
        :return:
        """
        for image in image_list:
            xref = image[0]  # 提取图片xref
            pix = fitz.Pixmap(pdf_image, xref)
            # 在当前目录创建一个子目录images，保存图片
            if not os.path.exists('./images'):
                os.makedirs('./images')
            if pix.n < 5:  # this is GRAY or RGB
                pix._writeIMG("./images/p%s-%s.png" % (index, xref), format=format, jpg_quality=jpg_quality)
            else:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1._writeIMG("./images/p%s-%s.png" % (index, xref), format=format, jpg_quality=jpg_quality)
                pix1 = None
            pix = None

    def __clear_images(self):
        """
        清空图片文件夹
        :return:
        """
        if os.path.exists('./images'):
            for file in os.listdir('./images'):
                os.remove(os.path.join('./images', file))
            os.rmdir('./images')


if __name__ == "__main__":
    c = CompareFile(name_distance="cosine", name_encode="word2vec")
    c.fit_transform("西交-应答文件可编辑版.pdf", "哈尔滨工程大学-标书-基于多元统计分析的多工况下典型传感器失效检测技术研究-终稿-正本.pdf")