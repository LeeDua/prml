import numpy as np
from scipy.stats import entropy as e_cal
from math import log2
from decision_tree.Data_schema import data,feature_cnt


#pass array
def cal_entrophy(array):
    value,counts = np.unique(array, return_counts=True)
    total_count = len(array)
    e = 0
    for i in range(len(value)):
        p = counts[i]/total_count
        if p == 0:
            return float("inf")
        else:
            e = e - p * log2(p)
    return e

def get_split_points(array):
    s_array = np.sort(np.unique(array))
    split_pts = []
    for i in range(len(s_array)-1):
        split_pts.append((s_array[i] + s_array[i+1])/2)
    return split_pts


#pass index list
def cal_info_gain(previous,partitioned_list):
    pre_e = cal_entrophy(extract_target(previous))
    total_partition_e = 0
    total_sample_cnt = len(previous)
    print("before partition:" + str(pre_e), end="")
    value, counts = np.unique(extract_target(previous), return_counts=True)
    print(value,counts)
    for partition in partitioned_list:
        partition_e = cal_entrophy(extract_target(partition))
        total_partition_e += partition_e * len(partition) / total_sample_cnt
    print("after partition:" + str(), end="")
    value, counts = np.unique(extract_target(partitioned_list[0]), return_counts=True)
    print(value,counts,end=" | ")
    value, counts = np.unique(extract_target(partitioned_list[1]), return_counts=True)
    print(value,counts)
    return pre_e - total_partition_e


def extract_data(index_list):
    d = data.data
    return d[index_list]


def extract_target(index_list):
    d = data.target
    return d[index_list]


def split_partition(previous,split_on,split_point):
    array_lt = []
    array_gt = []
    d = extract_data([i for i in range(150)])
    for item in previous:
        if d[item,split_on] >= split_point:
            array_gt.append(item)
        else:
            array_lt.append(item)
    return array_lt, array_gt


def get_split_criteria(index_list):
    split_points = []
    best_split_on = 0
    best_split_point = float("-inf")
    best_info_gain = float("-inf")
    print("splited_points:")
    for i in range(feature_cnt):
        pts = get_split_points(extract_data(index_list)[:,i])
        split_points.append(pts)
        print("length:" + str(len(pts)), end=" ")
        print(pts)
    print("trying out best info gain split")
    for i in range(feature_cnt):
        s_points = split_points[i]
        for split_point in s_points:
            splited_partion = split_partition(index_list,i,split_point)
            #print("splited partition on " + str(i) + " with value:" + str(split_point))
            #print(extract_data(splited_partion[0]))
            #print(extract_data(splited_partion[1]))
            info_gain = cal_info_gain(index_list,splited_partion)
            if info_gain >= best_info_gain:
                best_split_on = i
                best_split_point = split_point
                best_info_gain = info_gain
                print("better info gain found:" + str(info_gain))
    return best_split_on,best_split_point

