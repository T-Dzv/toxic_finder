import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
from transformers import BertTokenizer
import pandas as pd
import numpy as np 

#Добавляємо назви класів
class_names = [
    "Токсичність", "Сильна токсичність", "Непристойність", "Загрози", "Образи", "Ненавість до ідентичності"
    ]

#Заговолок та невеликий опис
st.title("Класифікація токсичності коментарів")
st.write("У сучасному світі соціальних мереж існує значна проблема токсичності в онлайн-коментарях, що створює негативне середовище для спілкування. Від зловживань до образ - це може призвести до припинення обміну думками та ідеями серед користувачів. Цей застосунок створений для класифікації коментарів на різні види токсичності.")

#Кнопка для візуалізації назв класів
st.write("Які є види токсичності?")
if st.button("Показати"):       
    x = 1
    for i in class_names:
        st.write(f'{x} - {i}')
        x+=1

#Форма для вводу коментаря
input_text = st.text_area("Перевірте ваш коментар")

# Функція очищення тексту
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

   # Ініціалізація токенізатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_attention_mask=True
    )
    return encoding['input_ids'].numpy().squeeze(0).tolist(), encoding['attention_mask'].numpy().squeeze(0).tolist()

if st.button('gfjdjg'):
    processed_text = clean_text(input_text)
    data = pd.DataFrame([[processed_text]], columns=["cleaned_comment_text"])
    data['input_ids'], data['attention_masks'] = zip(*data['cleaned_comment_text'].apply(
    lambda x: tokenize_text(x, tokenizer)))
    st.dataframe(data)
    mini_input_ids = tf.convert_to_tensor(data['input_ids'].apply(eval).tolist())
    mini_attention_masks = tf.convert_to_tensor(data['attention_masks'].apply(eval).tolist())
    mini_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': mini_input_ids,
    'attention_mask': mini_attention_masks}))
    # x_test = []
    # for x, y in mini_dataset.batch(1):
    #     x_test.append({
    #         'input_ids': x['input_ids'][0].numpy(),
    #         'attention_mask': x['attention_mask'][0].numpy()
    #     })

 