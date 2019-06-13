from decision_tree.Node import Node
from decision_tree.utils import *
from collections import deque

class Tree:
    def __init__(self, _max_depth, train_data):
        self.root = Node(None,False,train_data,True)
        self.current_node = self.root
        self.max_depth = _max_depth
        self.queue = deque([self.root])
        self.stack = [self.root]

    def re_init_queue(self):
        self.queue = deque([self.root])

    def re_init_stack(self):
        self.stack = [self.root]

    def build_tree(self):
        if len(self.queue) == 0:
            return
        self.current_node = self.queue.popleft()
        while self.current_node.current_depth<self.max_depth:
            child_is_leaf = False
            if self.current_node.current_depth >= self.max_depth - 1:
                child_is_leaf = True
            if self.current_node.is_leaf:
                self.build_tree()
                return
            else:
                node_left = Node(self.current_node,child_is_leaf,self.current_node.child_split[0],True)
                node_right = Node(self.current_node,child_is_leaf,self.current_node.child_split[1],False)
                self.current_node.add_child(node_left)
                self.current_node.add_child(node_right)
                self.queue.append(node_left)
                self.queue.append(node_right)
                self.build_tree()

    def predict(self,x_input):
        #print("\n-----start prediction------",end="")
        #print(x_input)
        self.current_node = self.root
        while not self.current_node.is_leaf:
            self.current_node = self.current_node.route_to_next(x_input)
        #print(self.current_node.predict())
        return self.current_node.predict()

    def print_tree(self):
        self.current_node = self.stack.pop()
        self.current_node.print_current()
        if len(self.current_node.child_list) != 0:
            self.stack.append(self.current_node.child_list[1])
            self.stack.append(self.current_node.child_list[0])
        if len(self.stack) != 0:
            self.print_tree()

    def test(self, test_set):
        test_data = extract_data(test_set)
        test_target = extract_target(test_set)
        total_cnt = len(test_set)
        correct_cnt = 0
        for i in range(total_cnt):
            label,name = self.predict(test_data[i])
            if label == test_target[i]:
                correct_cnt += 1
            else:
                print(test_data[i],end="->")
                print(test_target[i],end=",")
                print("miss-predict:" + str(label) + "(" + name + ")")
        print("Model accuracy:" + str(correct_cnt/total_cnt))
        return correct_cnt/total_cnt

    def post_prune(self):
        pass
