# _*_ coding: utf-8 _*_
# by:Snowkingliu
# 2017/11/3 下午3:53
import matplotlib.pyplot as plt


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    # plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plot_node(u'决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(u'叶子节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leafs(my_tree):
    """
    获取叶子节点的个数
    :param my_tree:
    :return:
    """
    num_leafs = 0
    first_str = my_tree.keys()[0]
    second_dic = my_tree[first_str]
    for key in second_dic.key():
        if type(second_dic[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dic[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    """
    获取树的深度
    :param my_tree:
    :return:
    """
    max_depth = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.key():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


if __name__ == '__main__':
    create_plot()
    # import treePlotter
    # treePlotter.createPlot0()
