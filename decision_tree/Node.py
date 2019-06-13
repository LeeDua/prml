from decision_tree.utils import *
from decision_tree.SelectCriteria import SelectCriteria
from decision_tree.Data_schema import feature_names
from decision_tree.Data_schema import target_names

class Node:
    def __init__(self,_parent,_is_leaf,_data_src,_is_left_node):
        self.parent  = _parent
        self.is_leaf = _is_leaf
        self.selectCriteria = None
        self.data_src = _data_src
        self.entrophy = cal_entrophy(extract_target(self.data_src))
        self.child_list = []
        self.child_split = None
        self.is_left_node = _is_left_node
        self.class_index = None
        self.class_name = None
        self.pruned = False
        if not self.parent:
            self.current_depth = 1
        else:
            self.current_depth = _parent.current_depth + 1
        self.check_is_leaf()
        if self.is_leaf:
            return
        else:
            self.get_select_criteria()

    def add_child(self,child_node):
        self.child_list.append(child_node)

    def predict(self):
        classes, counts = np.unique(extract_target(self.data_src), return_counts=True)
        class_cnt = len(classes)
        if class_cnt != 0:
            self.class_index = classes[counts.tolist().index(max(counts))]
            self.class_name = target_names[self.class_index]
        return self.class_index,self.class_name

    def print_current(self):
        for i in range(self.current_depth):
            print("--|",end="")
        if not self.is_leaf:
            print(feature_names[self.selectCriteria.attri_index],end="")
            print("<", end="")
            print(self.selectCriteria.threshold_value,end=" ")
        else:
            print("[leaf->" + self.class_name + "]", end=" ")
        print(str(len(self.data_src)) + "  " + str(self.entrophy) + "  ", end="")
        print(self.data_src)

    def get_select_criteria(self):
        #print("-----Debugging get criteria------")
        #for i in range(self.current_depth):
        #    print("--|", end="")
        #print(str(self.current_depth))
        #print("data_src:",end="")
        #print(self.data_src)
        best_split_on,best_split_point = get_split_criteria(self.data_src)
        self.selectCriteria = SelectCriteria(best_split_on,best_split_point)
        self.child_split = split_partition(self.data_src,best_split_on,best_split_point)
        #print("-------------------------------\n")

    def check_is_leaf(self):
        classes,counts = np.unique(extract_target(self.data_src),return_counts=True)
        class_cnt = len(classes)
        if class_cnt <= 1:
            self.is_leaf = True
        if class_cnt != 0:
            self.class_index = classes[counts.tolist().index(max(counts))]
            self.class_name = target_names[self.class_index]
            '''
            if class_cnt == 1:
                self.class_index = classes[0]
                self.class_name = target_names[self.class_index]
            else:
                self.class_name = "ERROR"
        else:
            self.class_index = classes[counts.tolist().index(max(counts))]
            self.class_name = target_names[self.class_index]
        '''
    def route_to_next(self,x_input):
        if self.is_leaf:
            print("Should not call route next on leaf node")
            return
        if x_input[self.selectCriteria.attri_index] < self.selectCriteria.threshold_value:
            #print(str(x_input) + "->" + str(x_input[self.selectCriteria.attri_index]) + "<" + str(self.selectCriteria.threshold_value))
            return self.child_list[0]
        else:
            #print(str(x_input) + "->" + str(x_input[self.selectCriteria.attri_index]) + ">=" + str(self.selectCriteria.threshold_value))
            return self.child_list[1]

