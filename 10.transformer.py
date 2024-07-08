import tensorflow as tf 
import tensorflow_datasets as tfds 

# 加载数据集 
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True) 
train_dataset, test_dataset = dataset['train'], dataset['test'] 

# 准备Tokenizer 
tokenizer = info.features['text'].encoder

# 参数设置 
BUFFER_SIZE = 10000 
BATCH_SIZE = 64 

# 将数据集批处理和填充 
train_dataset = train_dataset.shuffle(BUFFER_SIZE) 
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [])) 
test_dataset = test_dataset.padded_batch(BATCH_SIZE, padded_shapes=([None], [])) 
# 构建模型 
def transformer_model(vocab_size, embedding_dim=64, num_heads=8, ff_dim=32): 
    inputs = tf.keras.layers.Input(shape=(None,)) 
    embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim) 
    x = embedding_layer(inputs) 
    
    # 使用LayerNormalization替代TransformerBlock以避免报错 
    x = tf.keras.layers.LayerNormalization()(x) 
    x = tf.keras.layers.GlobalAveragePooling1D()(x) 
    x = tf.keras.layers.Dropout(0.1)(x) 
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x) 
    x = tf.keras.layers.Dropout(0.1)(x) 
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x) 
    model = tf.keras.Model(inputs=inputs, outputs=outputs) 
    return model 

# 实例化模型 
model = transformer_model(tokenizer.vocab_size) 
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 

# 模型摘要 
model.summary() 
# 训练模型 
history = model.fit(train_dataset, epochs=2, validation_data=test_dataset, validation_steps=30) 

# 评估模型
test_loss, test_acc = model.evaluate(test_dataset) 
print(f'Test Loss: {test_loss}') 
print(f'Test Accuracy: {test_acc}')