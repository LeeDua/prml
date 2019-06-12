from sklearn import datasets
from sklearn.tree import *
import matplotlib.pyplot as plt
import graphviz
import os
os.environ["PATH"] += os.pathsep + r'D:\AppSource\anaconda\Library\bin\graphviz'


iris = datasets.load_iris()

dtc = DecisionTreeClassifier()
dtc = dtc.fit(iris.data, iris.target)

#tree.plot_tree(dtc)
#plt.show()


def graph_visualization():
    visualization = export_graphviz(dtc,
                                         out_file="tree_visual.pdf",
                                         feature_names=iris.feature_names,
                                         class_names=iris.target_names,
                                         filled=True, rounded=True,
                                         special_characters=True)
    graph = graphviz.Source(visualization)
    graph.render("iris")


def text_visualizaiton():
    text_visualization = export_text(dtc,feature_names=iris['feature_names'])
    print(text_visualization)

