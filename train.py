import tensorflow as tf
from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

data = pd.read_csv('bot_data.csv', sep=';')
questions = data['Вопрос'].values
answers = data['Ответ'].values


def get_token_lengths(texts, tokenizer):
    lengths = [len(tokenizer.encode(text)) for text in texts]
    return lengths


question_lengths = get_token_lengths(questions, tokenizer)
answer_lengths = get_token_lengths(answers, tokenizer)

max_question_length = max(question_lengths)
max_answer_length = max(answer_lengths)

max_length = max(max_question_length, max_answer_length) + 50

dataset = []
for i in range(len(questions)):
    q_inputs = tokenizer([questions[i]], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    a_inputs = tokenizer([answers[i]], padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    q_emb = bert_model(**q_inputs)['last_hidden_state'][:, 0, :].detach().numpy()
    a_emb = bert_model(**a_inputs)['last_hidden_state'][:, 0, :].detach().numpy()

    dataset.append([np.array(q_emb[0]), np.array(a_emb[0])])

dataset = np.array(dataset)

X, Y = [], []
for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0]):
        X.append(np.concatenate([dataset[i, 0, :], dataset[j, 1, :]], axis=0))
        Y.append(1 if i == j else 0)

X = np.array(X)
Y = np.array(Y)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(512, activation='selu'),
    tf.keras.layers.Dense(256, activation='selu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

es = tf.keras.callbacks.EarlyStopping(monitor='auc', mode='max', patience=10, restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])

model.fit(X, Y, epochs=350, class_weight={0:1, 1:np.sqrt(Y.shape[0])-1})
model.save('bot_model.keras')
