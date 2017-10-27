# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2017/10/24 10:44
from os import listdir

from numpy import *
import operator
import kNN as knn
import matplotlib
import matplotlib.pyplot as plt


def create_date_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify0(in_x, data_set, labels, k):
    """
    K-近邻算法
    :param in_x:被分类给予标签预测的向量
    :param data_set:训练样本集
    :param labels:训练样本对应的标签
    :param k:选择最近的k个点，即参考的排序后的k个较近的点来统计
    :return:权重最大的标签
    """
    # 获取行数，shape的0项是行数，1项是列数
    data_set_size = data_set.shape[0]
    # tile是重复函数，第二个参数可以是一个数也可以是一个元组，若是一个数代表列重复的次数；元组的0项是代表列重复次数，1项代表行重复次数
    # diffMat是inX与dataSet每一个dataSet的差组成的矩阵
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    # 获取各项的差的平方
    sq_diff_mat = diff_mat ** 2
    # 若无axis参数代表全部相加，axis=0代表按照列相加，axis=1代表按照行相加，返回总数一个行向量（一维数组）
    sq_disatance = sq_diff_mat.sum(axis=1)
    # 开根号
    distance = sq_disatance ** 0.5
    # argsort是获取矩阵从小到大的下标号,若前面是一个矩阵,axis=0代表按照行排序返回按照行排序，axis=1代表按照列排序，
    # 当前面是1维矩阵时，axis=0可省略
    # 这里代表获取最小的序列
    sorted_dist_indicies = distance.argsort()
    class_count = {}
    # 取k个值
    for i in range(k):
        # sorted_dist_indicies是个array，只需[i]就可以得到对应到数组里的下标
        vote_lable = labels[sorted_dist_indicies[i]]
        # 让该标签的权重加大
        class_count[vote_lable] = class_count.get(vote_lable, 0) + 1
    # sorted返回一个排序后序列，itemgetter代表按照value来排序，keys按照key来排序，
    # key=operator.itemgetter(1)表示保留默认即原有的key作为key，0代表的索引值
    # reverse=True代表降序，reverse=False代表升序
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(0), reverse=True)
    # 将权重最大的标签返回出来
    return sorted_class_count[0][0]


def file2matrix(filename):
    """

    :param filename: 文件的路径/文件名
    :return:NumPy的矩阵，对应标签
    """
    # 打开并读取文件
    fr = open(filename)
    array_lines = fr.readlines()
    # 获取文件的行数
    number_of_lines = len(array_lines)
    # zeros传入一个元组，返回指定形状的用0来填充的矩阵
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_lines:
        # strip()为空代表清除字符串两端的"\t、\n等"
        line = line.strip()
        list_from_line = line.split('\t')
        # index代表return_mat的行数，[0: 3]取前三个
        return_mat[index, :] = list_from_line[0: 3]
        # 生成标签列表
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def auto_norm(data_set):
    """
    归一化特征值
    :param data_set:已经numpy化了的矩阵
    :return:
    """
    # max 或 min 函数求最值。括号不加参数代表矩阵的最大的元素，若传0代表求各列的最大值，返回是一个行向量，代表每一列中的最大/小值
    # 若传1代表求各行的最大/小值，返回是一个行向量，代表每一行中的最大值
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # shape获取段数
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    """
    错误率
    :return:
    """
    ho_ratio = 0.03
    dating_data_mat, dating_labels = file2matrix("file/datingTestSet2.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 获取行数，shape的0项是行数，1项是列数
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs: m], 3)
        print "分类器获得的结果是：%d，其中真实的结果是：%d个" %(classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print "错误率是：%f" % (error_count / float(num_test_vecs))


def classify_person():
    """
    判断对某种类型的人的喜欢程度
    :return:
    """
    result_list = ['完全没兴趣', '有一点点考虑', '很感兴趣']
    percent_tats = float(raw_input("打电动的时间占比是多少呢？"))
    ff_miles = float(raw_input("每年飞行里程是多少呢？"))
    ice_cream = float(raw_input("每年吃多少升的冰淇淋🍦呢？"))
    dating_data_mat, dating_labels = file2matrix("file/datingTestSet2.txt")
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat,dating_labels, 3)
    print "你对类似于这样的人会：", result_list[classifier_result - 1]


# 手写识别系列
def img2vector(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    hw_labels = []
    # 训练文件目录
    training_file_list = listdir("file/digits/trainingDigits")
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        # 获取文件名
        file_name_str = training_file_list[i]
        # 获得去掉后缀名部分文件名
        file_str = file_name_str.split('.')[0]
        # 获取数字
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        # 读取文件
        training_mat[i, :] = img2vector("file/digits/trainingDigits/%s" % file_name_str)
    # 测试文件目录
    test_file_list = listdir("file/digits/testDigits")
    # 错误计数
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        # 获得去掉后缀名部分文件名
        file_str = file_name_str.split('.')[0]
        # 获取数字
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector("file/digits/testDigits/%s" % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print "分类的结果是：%d，真实的类别是：%d" % (classifier_result, class_num_str)
        if classifier_result != class_num_str:
            error_count += 1.0
        print "错误分类的数量是：%d" % error_count
        print "错误率是：%f" % (error_count / float(m_test))


if __name__ == '__main__':
    # dating_data_mat, class_label_vector = file2matrix("file/datingTestSet2.txt")
    # print create_date_set()
    # # 函数创建的窗口
    # fig = plt.figure()
    # # “111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”
    # ax = fig.add_subplot(111)
    # # scatter函数原型函数
    # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
    # 15.0*array(class_label_vector), 15.0*array(class_label_vector))
    # classify_person()
    # 错误率
    # dating_class_test()
    # test_vector = img2vector("file/digits/testDigits/0_13.txt")
    # print test_vector[0, 0:31]
    handwriting_class_test()

