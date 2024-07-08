import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#创建数据集
X = np.array([[1], [2], [3], [4], [5]]) #房间数量
y = np.array([1, 2, 3, 3.5, 4]) #房价，单位：百万

#切分数据集
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

#模型训练
model = LinearRegression()
model.fit(X_train, y_train)

#预测
y_pred = model.predict(X_test)

#可视化
plt.scatter(X, y, color='blue') #原始数据
plt.plot(X, model.predict(X), color='red') #拟合线
plt.xlabel('Rooms')
plt.ylabel('Price (millions)')
plt.show()

#打印预测值和实际值
print(f'Predicted prices: {y_pred}')
print(f'Actual prices: {y_test}')