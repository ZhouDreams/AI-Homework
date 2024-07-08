import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 步骤3：生成数据集
# 生成正弦波数据
t = np.linspace(0, 100, 1000)
data = np.sin(t)
# 准备输入输出数据（假设使用前5个点预测下一个点）

X, y = [], []
for i in range(len(data) - 5):
    X = np.append(X,data[i:i+5])
    y = np.append(y,data[i+5])
    X, y = np.array(X), np.array(y)
    X = np.expand_dims(X, axis=2) # RNN需要三维输入 [samples, timesteps, features]

# 步骤4：构建RNN模型
model = Sequential([
SimpleRNN(50, activation='relu', input_shape=(5, 1)),
Dense(1)
])

# 步骤5：编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 步骤6：训练模型
model.fit(X, y, epochs=200, verbose=1)

# 步骤7：模型预测
# 假设使用最后5个点进行预测
test_input = data[-5:]
test_input = test_input.reshape((1, 5, 1))
predicted = model.predict(test_input, verbose=0)
print(f'预测的下一个点的值: {predicted}')