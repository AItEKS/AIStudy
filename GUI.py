import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd


def load_model_and_data():
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

    data = pd.read_csv('bot_data.csv', sep=';')
    questions = data['Вопрос'].values
    answers = data['Ответ'].values

    answer_embeddings = compute_answer_embeddings(answers, tokenizer, bert_model)

    return tokenizer, bert_model, questions, answers, answer_embeddings


def compute_answer_embeddings(answers, tokenizer, bert_model):
    answer_embeddings = []
    for answer in answers:
        a_inputs = tokenizer([answer], padding=True, truncation=True, max_length=512, return_tensors="pt")
        a_emb = bert_model(**a_inputs)['last_hidden_state'][:, 0, :].detach().numpy()
        answer_embeddings.append(a_emb[0])
    return np.array(answer_embeddings)


def predict_answer(new_question, answer_embeddings, model):
    new_q_inputs = tokenizer([new_question], padding=True, truncation=True, max_length=512, return_tensors="pt")
    new_q_emb = bert_model(**new_q_inputs)['last_hidden_state'][:, 0, :].detach().numpy()

    X_new = np.concatenate([np.tile(new_q_emb, (len(answers), 1)), answer_embeddings], axis=1)
    predictions = model.predict(X_new)

    best_match_index = np.argmax(predictions)
    return answers[best_match_index], predictions[best_match_index]


def get_answer():
    question = question_entry.get()
    if question:
        best_answer, confidence = predict_answer(question, answer_embeddings, model)

        if isinstance(confidence, np.ndarray):
            confidence = confidence.item()

        messagebox.showinfo("Ответ", f"Лучший ответ: {best_answer}")
    else:
        messagebox.showwarning("Внимание", "Пожалуйста, введите вопрос.")


if __name__ == '__main__':
    tokenizer, bert_model, questions, answers, answer_embeddings = load_model_and_data()

    model = tf.keras.models.load_model('bot_model.keras')

    root = tk.Tk()
    root.title("Вопрос-Ответ Бот")

    tk.Label(root, text="Введите ваш вопрос:").pack(pady=10)
    question_entry = tk.Entry(root, width=50)
    question_entry.pack(pady=10)

    submit_button = tk.Button(root, text="Получить ответ", command=get_answer)
    submit_button.pack(pady=20)

    root.mainloop()