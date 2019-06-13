from decision_tree.Tree import Tree
from decision_tree.Data_schema import data_len
from random import shuffle
import statistics

test_epoch = 20

max_depth_list = [i for i in range(2,10)]
max_depth_comparision = dict()

for max_depth in max_depth_list:
    accuracy = []
    for epoch in range(test_epoch):
        all_data = [i for i in range(data_len)]
        shuffle(all_data)
        proportion = 0.6
        train_set = all_data[0:round(data_len*proportion)]
        test_set = all_data[round(data_len*proportion) + 1:]

        decision_tree = Tree(max_depth, train_set)
        decision_tree.build_tree()
        decision_tree.print_tree()
        #decision_tree.predict([5.9, 3. , 5.1, 1.8])
        a = decision_tree.test(test_set)
        accuracy.append(a)
        print("--------------------")
    print("average accuracy:", end="")
    mean_acc = statistics.mean(accuracy)
    print(mean_acc)
    max_depth_comparision[str(max_depth)] = mean_acc

print(max_depth_comparision)

