from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names
data_len = len(y)
feature_cnt = len(data.data[0])
