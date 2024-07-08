import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted
        
    def activation_function(self, x):
        return np.where(x>=0, 1, 0)

# 示例数据和标签
X = np.array([[2, 3], [1, 2], [2, 1], [3, 2]])
y = np.array([0, 0, 1, 1]) # 二分类标签

# 创建感知机实例
perceptron = Perceptron(learning_rate=0.01, n_iter=10)

# 训练模型
perceptron.fit(X, y)

# 预测新的数据点
predictions = perceptron.predict(X)
print("Predictions:", predictions)
