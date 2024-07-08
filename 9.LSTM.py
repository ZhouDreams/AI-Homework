import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 生成数据集
def generate_sequence(length=100):
    return np.arange(length)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

sequence = generate_sequence(100)
n_steps = 5
X, y = split_sequence(sequence, n_steps)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# 构建LSTM模型
model = Sequential([
LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X, y, epochs=200, verbose=0)

# 进行预测
x_input = np.array([95, 96, 97, 98, 99])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(f'Predicted: {yhat}')