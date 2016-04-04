# -*- coding:utf8 -*-
"""
非剪枝ID3算法的python实现
本模块是博主依据ID3思想的一个简单coding，用于深入理解ID3
ID3算法已经有python包：http://id3-py.sourceforge.net/

参考文档
《数据挖掘-实用机器学习技术》Witten,I.H. p65
http://zc0604.iteye.com/blog/1462825
http://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
http://www.cnblogs.com/superhuake/archive/2012/07/25/2609124.html
http://my.oschina.net/dfsj66011/blog/343647
http://blog.csdn.net/leeshuheng/article/details/7777722
http://blog.sina.com.cn/s/blog_6e85bf420100ohma.html
http://blog.csdn.net/pi9nc/article/details/12197119
"""
from math import log
from copy import deepcopy


def _entro(D, A=[], tag_column=-1):
    """
    _entro(D, A=[], tag_column=-1)
    熵和条件熵计算

    input:
    @D          数据集
    @tag_column 标签列，默认-1
    @A          条件列，为空则计算熵，不为空则计算在该列(s)条件下的熵

    output:
    熵值 float
    """
    tags = [line[tag_column] for line in D]  # 存储所有标签
    attribs = []  # 存储所有条件
    attribs_and_tags = []  # 存储条件列和标签
    A = deepcopy(A)
    A.append(tag_column)
    # 解析D为需要的格式
    for line in D:
        attribs.append(tuple([line[i] for i in A if i != tag_column]))
        attribs_and_tags.append(tuple([line[i] for i in A]))

    distinct_tags = set(tags)  # 标签去重
    attrib_count = {}  # 存储条件列：总权重，按标签分开的权重
    temp_dict = {'count': 0, 'tags': {}.fromkeys(distinct_tags, 0)}
    for attr in attribs:
        if attr not in attrib_count:
            attrib_count[attr] = deepcopy(temp_dict)
        attrib_count[attr]['count'] += + 1
    for attr in attrib_count:
        for tag in distinct_tags:
            attrib_count[attr]['tags'][tag] = attribs_and_tags.\
                count(attr + (tag,))

    # 计算熵
    condition_entro = 0  # 熵或条件熵
    len_D = float(len(D))  # 数据集大小
    for attrib, detail in attrib_count.items():
        part_entro = 0  # 条件列的熵
        count = float(detail['count'])
        for tag, tag_c in detail['tags'].items():
            part_entro -= (tag_c / count) * log((tag_c / count), 2) \
                if (tag_c / count) else 0
        condition_entro += part_entro * count / len_D  # 加权求和
    return condition_entro  # 返回熵

# 计算条件熵的差值
_gain = lambda D, oldA, newA: _entro(D, oldA) - _entro(D, newA)


def best_attr(D, oldA=[], threshold_gain=0):
    """
    best_attr(D, oldA=[], threshold_gain=0)
    在条件oldA情况下寻找最优的下一个条件

    input:
    @D              数据集
    @oldA           原始条件
    @threshold_gain 系统阈值

    output:
    {'attr': 新条件, 'gain': 原始条件与新条件的熵差, 'entro': 新熵}
    """
    _best_attr = {'attr': None, 'gain': 0, 'entro': None}
    attr_num = len(D[0]) - 1  # 条件列计数
    for attr in range(attr_num):
        if attr in oldA:
            continue
        newA = deepcopy(oldA)
        newA.append(attr)
        resp = _gain(D, oldA, newA)
        if resp > _best_attr['gain']:
            _best_attr['attr'] = attr
            _best_attr['gain'] = resp
            _best_attr['entro'] = _entro(D, newA)
    return _best_attr


def best_tag(tags):
    """
    best_tag(tags)
    找出出现最多的标签
    """
    tag_count = {}.fromkeys(set(tags), 0)
    for tag in tags:
        tag_count[tag] += 1
    _best_tag = {'tag': None, 'count': 0}
    for tag, count in tag_count:
        if count > _best_tag['count']:
            _best_tag['tag'] = tag
            _best_tag['count'] = count
    return _best_tag


def split_D_by_A(D, A=[], tag_column=-1):
    """
    split_D_by_A(D, A=[], tag_column=-1)
    根据A划分数据集D

    output:
    {attr1: sub_D1, attr2: sub_D2, ...}
    """
    A = [A] if not isinstance(A, list) else A
    attribs_and_tags = {}
    for line in D:
        temp_D = []
        for index, i in enumerate(line):
            temp_D.append(i) if index not in A else None
        key = tuple([line[i] for i in A if i != tag_column])
        if key not in attribs_and_tags:
            attribs_and_tags[key] = []
        attribs_and_tags[key].append(temp_D)
    return attribs_and_tags


def id3_tree(D, title=None, tag_column=-1):
    """
    id3_tree(D, title=None, tag_column=-1)
    根据数据集构造树
    @D 数据集
    @title<list> 列名
    """
    tags = [line[tag_column] for line in D]
    # 只有一个类别时不再划分
    if tags.count(tags[0]) == len(tags):
        return tags[0]

    # 没有条件可以划分时，返回出现最多的标签
    if len(D[0]) == 1:
        return best_tag(tags)

    b_attr = best_attr(D, [])['attr']
    this_title = title[b_attr]
    tree = {this_title: {}}
    del title[b_attr]
    sub_Ds = split_D_by_A(D, b_attr)
    for attr, sub_D in sub_Ds.items():
        try:
            tree[this_title][attr[0]] = id3_tree(sub_D, title)
        except:
            raise
    return tree


if __name__ == '__main__':
    from data import dataset1
    print id3_tree(dataset1['data'], dataset1['title'])
