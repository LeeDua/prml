import numpy as np
from PerceptionMachine.data_supply import *
from matplotlib import pyplot as plt


class PerceptionMachine:
    def __init__(self,_learning_rate,_X,_y,_max_iteration = 50000):
        self.learning_rate = _learning_rate
        self.max_iteration = _max_iteration
        self.X = _X
        self.y = _y
        self.W = np.random.rand(2)
        self.b = np.random.rand(1)
        self.iter_count = 0
        self.x1_max = np.max([x[0] for x in X]) + .2
        self.x2_max = np.max([x[1] for x in X]) + .2
        self.x1_min = np.min([x[0] for x in X]) - .2
        self.x2_min = np.min([x[1] for x in X]) - .2
        self.point_ajusted = 0

    def predict(self,X_pre):
        return np.dot(self.W, X_pre) + self.b

    def train(self):
        flag = True
        while flag:
            self.iter_count += 1
            flag = False
            for i in range(len(self.y)):
                p = self.predict(self.X[i])
                delta_w = self.predict(self.X[i])*self.y[i]
                if delta_w < 0:
                    #print(p, self.y[i],delta_w)
                    self.W += self.learning_rate * self.X[i] * self.y[i]
                    self.b += self.learning_rate * self.y[i]
                    self.point_ajusted += 1
                    flag = True
                    #break
            #if self.iter_count % 5000 == 0:
              #  self.plot_current()
            if self.iter_count == self.max_iteration:
                print("fail to converge before max_iter")
                break
                #exit(-1)
        #print("training done")
        #print(self.iter_count)
        self.plot_current()

    def plot_current(self):
        plt.title(str(self.iter_count))
        #print(self.W, self.b)
        x1 = np.arange(self.x1_min, self.x1_max, 0.01)
        x2 = (-self.b - self.W[0]*x1)/self.W[1]
        plt.plot(x1,x2)
        for i in range(len(self.y)):
            f = 'x'
            if self.y[i] == -1:
                f = 'o'
            plt.plot(self.X[i][0],self.X[i][1],format(f))
        plt.show()



if __name__ == '__main__':
    X, y = get_samples()
    model = PerceptionMachine(0.1, X, y)
    model.train()
    exit(-1)

    learning_rate_list = [0.001,0.003,0.01,0.03,0.1,0.3,0.9]
    try_count_for_each_lr = 50
    record = dict()
    for lr in learning_rate_list:
        record[str(lr)] = []
    for i in range(len(learning_rate_list)):
        learning_rate = learning_rate_list[i]
        for t in range(try_count_for_each_lr):
            X,y = get_samples()
            model = PerceptionMachine(learning_rate,X,y)
            model.train()
            record[str(learning_rate)].append((model.iter_count,model.point_ajusted))
            print("trying:" +str(learning_rate) + " loop " + str(t) + "--iter count " + str(model.iter_count))
    print(record)
    for r in record.keys():
        print(r,np.average([x[0] for x in record[r]]),np.average([x[1] for x in record[r]]))





