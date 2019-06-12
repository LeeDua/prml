from decision_tree.Node import Node
from decision_tree.utils import *
from collections import deque
from decision_tree.Data_schema import data_len

class Tree:
    def __init__(self, _max_depth):
        self.root = Node(None,False,[i for i in range(data_len)],True)
        self.current_node = self.root
        self.max_depth = _max_depth
        self.queue = deque([self.root])
        self.stack = [self.root]

    def build_tree(self):
        if len(self.queue) == 0:
            return
        self.current_node = self.queue.popleft()
        while(self.current_node.current_depth<self.max_depth):
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

    def predict(self):
        pass

    def print_tree(self):
        self.current_node = self.stack.pop()
        self.current_node.print_current()
        if len(self.current_node.child_list) != 0:
            self.stack.append(self.current_node.child_list[1])
            self.stack.append(self.current_node.child_list[0])
        if len(self.stack) != 0:
            self.print_tree()
