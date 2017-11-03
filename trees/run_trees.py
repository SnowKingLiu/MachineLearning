# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2017/11/1 下午5:35

from math import log
import operator


def calc_shannon_ent(data_set):
    """
    计算给定数据的香农熵
    :param data_set:
    :return:
    """
    # 获取数据集的个数
    num_entries = len(data_set)
    label_counts = {}
    # 得到所有结果的频数
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 香农熵
    shannon_ent = 0.0
    for key in label_counts:
        # 获取该分类的概率
        prob = float(label_counts[key]) / num_entries
        # 熵等于信息增益的和
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_data_set():
    """
    造demo数据
    :return:
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['不浮出水面可生存否', '有脚蹼否']
    return data_set, labels


def split_data_set(data_set, axis, value):
    """
    按照给定特征值划分数据集，就是选出第axis项为value的，出去掉第axis后的结果
    :param data_set:
    :param axis:
    :param value:
    :return:
    """
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 剔除axis项
            reduced_feat_vec = feat_vec[:axis]
            # extend要解包一项项插入的
            reduced_feat_vec.extend(feat_vec[axis+1:])
            # append直接将()里的最为一项插入
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选出熵最大，即影响最大的属性
    :param data_set:
    :return:
    """
    # 参考条件的个数
    num_features = len(data_set[0]) - 1
    # 香农熵值
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 第i项的值列表
        feat_list = [example[i] for example in data_set]
        # 去重得到分类标签
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            # 获取所有类别所有可能值包含的信息期望值
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        # 信息增益等于集合的信息熵 - 该属性的期望熵
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_lsit):
    class_count = {}
    for vote in class_lsit:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    递归地获取一棵决策树
    :param data_set:
    :param labels:
    :return:
    """
    # 获取类别列表
    class_list = [example[-1] for example in data_set]
    # 若都是同一种类型，直接返回该类型
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 若数据集只剩结果，即，没有参数了
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 获取最关键的属性
    best_feat = choose_best_feature_to_split(data_set)
    # 得到最重要的属性的名称
    best_feat_label = labels[best_feat]
    # 创建树🌲
    my_tree = {best_feat_label: {}}
    # 将该名称从labels中删除掉
    del(labels[best_feat])
    # 获取到该属性的情况
    feat_values = [example[best_feat] for example in data_set]
    # 去重
    unique_vals = set(feat_values)
    for value in unique_vals:
        # 深拷贝
        sub_labels = labels[:]
        # 对分类为value的再创建子树🌲
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


if __name__ == '__main__':
    my_data_set, my_labels = create_data_set()
    # print my_data_set
    # print calc_shannon_ent(my_data_set)
    # print split_data_set(my_data_set, 0, 1)
    # print choose_best_fearure_to_split(my_data_set)
    print create_tree(my_data_set, my_labels)

