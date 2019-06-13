from decision_tree.Tree import Tree
from decision_tree.Data_schema import data_len
from random import shuffle
import statistics

test_epoch = 50
proportion_list = [0.5,0.6,0.7,0.8,0.9]
#proportion_list = [0.5,0.6]
prune_at_once = True

whole_tree = Tree(100, [i for i in range(data_len)])
whole_tree.build_tree()
#whole_tree.print_tree()
whole_tree_depth = whole_tree.actual_max_depth

#max_depth_list = [10]
max_depth_list = [i for i in range(2,whole_tree_depth+1)]

proportion_comparision = dict()

for proportion in proportion_list:
    max_depth_comparision = dict()
    for max_depth in max_depth_list:
        accuracy = []
        pruned_accuracy = []
        for epoch in range(test_epoch):
            all_data = [i for i in range(data_len)]
            shuffle(all_data)
            train_set = all_data[0:round(data_len*proportion)]
            test_set = all_data[round(data_len*proportion) + 1:]

            decision_tree = Tree(max_depth, train_set)
            decision_tree.build_tree()
            decision_tree.print_tree()
            #decision_tree.predict([5.9, 3. , 5.1, 1.8])
            a = decision_tree.test(test_set)
            accuracy.append(a)
            print("------------pruning-------------")
            decision_tree.post_prune(test_set,prune_at_once)
            print("------------pruned tree---------")
            decision_tree.re_init_stack()
            decision_tree.print_tree()
            pruned_accuracy.append(decision_tree.test(test_set))
            print("--------------------")
        print("average accuracy:", end="")
        mean_acc = statistics.mean(accuracy)*100
        print(mean_acc)
        print("average pruned accuracy:", end="")
        mean_p_acc = statistics.mean(pruned_accuracy)*100
        print(mean_p_acc)
        max_depth_comparision[str(max_depth)] = (mean_acc,mean_p_acc,mean_p_acc-mean_acc)
    proportion_comparision[str(proportion)] = max_depth_comparision

print("--------result----------")
print("epoch on each model:" + str(test_epoch))
print("whole_tree_depth:" + str(whole_tree_depth))
print("prune at one:" + str(prune_at_once))
print("------------------------")
for key,value in proportion_comparision.items():
    print("proportion=" + str(key))
    for k,v in value.items():
        print("    ",end="")
        print(k,end=" ")
        print(v)

