import numpy as np
import collections


def get_samples():
    max_try = 50000
    try_count = max_try
    pos_count = 0
    neg_count = 0
    while pos_count < 50 or neg_count < 50:
        try_count += 1
        if try_count >= max_try:
            try_count = 0
            X = []
            y = []
            target_w = np.random.rand(2)
            target_b = np.random.rand(1)
            threshold = target_b / 5
            pos_count = 0
            neg_count = 0
        x = (np.random.rand(2) - 0.5)
        if np.dot(target_w, x) + target_b > threshold:
            if pos_count < 50:
                pos_count += 1
                X.append(x)
                y.append(1)
        elif np.dot(target_w,x) + target_b < - threshold:
            if neg_count < 50:
                neg_count += 1
                X.append(x)
                y.append(-1)
    #print(target_w,target_b)
    index = [i for i in range(100)]
    np.random.shuffle(index)
    X_s = [X[i] for i in index]
    y_s = [y[i] for i in index]
    #print(X_s)
    #print(y_s)
    return X_s, y_s
