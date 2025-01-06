import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
from transformers import BertTokenizer
import pandas as pd
import numpy as np 
from transformers import TFBertForSequenceClassification
from tensorflow.keras.utils import custom_object_scope

# Функція очищення тексту
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text= re.sub(r'[а-яА-ЯёЁІіҐґЄєЇї]','', text)
    text = text.strip()
    return text

#Функція токенізації тексту
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

#Функція обробки тексту для моделі 
def processing_text(tokenizer, tokenize_text, processed_text):
    df = pd.DataFrame([[processed_text]], columns=["cleaned_comment_text"])
    df['input_ids'], df['attention_masks'] = zip(*df['cleaned_comment_text'].apply(
    lambda x: tokenize_text(x, tokenizer)))
    for column in ['input_ids', 'attention_masks']:
        df[column] = df[column].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))
    input_ids = np.stack(df['input_ids'].values)
    attention_mask = np.stack(df['attention_masks'].values)
    return input_ids,attention_mask

#Функція передбачень моделі
def predict(model, input_ids, attention_mask):
    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
    probs = tf.sigmoid(predictions).numpy()[0]
    return predictions,probs

#Функція виводу передбачень
def show_predict(LABELS, predictions, probs):
    present_classes = [LABELS[i] for i, prob in enumerate(probs) if prob > 0.5]  
    if present_classes:
        st.write("Ваш коментар класифікований як:")
        for label in present_classes:
            st.write(f"- **{label}**")
    else:
        st.write("Ваш коментар класифікований як:")
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = LABELS[predicted_class[0]]
        st.write(f"- **{predicted_label}**")

#Добавляємо назви класів
LABELS = ['Токсичний', 'Сильно токсичний', 'Непристойний', 'Погрози', 'Образи', 'Ненависть до ідентичності', 'Не токсичний']

#Загружаємо модель
custom_objects = {'TFBertMainLayer': TFBertForSequenceClassification}
with custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('model-5.h5')  

# Ініціалізація токенізатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Заговолок та невеликий опис
st.title("Класифікація токсичності коментарів")
st.write("У сучасному світі соціальних мереж існує значна проблема токсичності в онлайн-коментарях, що створює негативне середовище для спілкування. Від зловживань до образ - це може призвести до припинення обміну думками та ідеями серед користувачів. Цей застосунок створений для класифікації коментарів на різні види токсичності.")

#Кнопка для візуалізації назв класів
st.write("Які є види токсичності?")
if st.button("Показати"):       
    x = 1
    for i in LABELS:
        st.write(f'{x} - {i}')
        x+=1

#Вибір способу вводу текста
select = st.selectbox('Виберіть спосіб передачі тексту', ['.', 'Вести вручну', 'Завантажити файл'])

if select=='Вести вручну':
    input_text = st.text_area("Введіть ваш коментар")
    if st.button('Передбачити'):
        processed_text = clean_text(input_text)
        if processed_text=="":
            st.error('Будь ласка, перевірте ваш коментар. Він має бути написаним англійською мовою.')
        else:
            input_ids, attention_mask = processing_text(tokenizer, tokenize_text, processed_text)
            predictions, probs = predict(model, input_ids, attention_mask) 
            show_predict(LABELS, predictions, probs)
elif select=='Завантажити файл':
    file = st.file_uploader('Загрузіть ваш файл. Файл має бути у форматі txt. Кожен коментар має починатися з нового рядка', ['txt'])
    if file:
        text = file.read().decode("utf-8")
        texts = [t.strip() for t in text.split("\n") if t.strip()]
        if texts and len(texts) > 5:
            st.error("Будь ласка, перевірте ваш файл. Коментарів має бути не більше 5")
        else:
            if st.button('Передбачити'):
                for i, t in enumerate(texts):
                    processed_text = clean_text(t)
                    if processed_text=="":
                        st.error("Будь ласка, перевірте ваш файл. Всі коментарі мають бути з нового рядка та написані англійською мовою.")
                    else:
                        input_ids, attention_mask = processing_text(tokenizer, tokenize_text, processed_text)
                        predictions, probs = predict(model, input_ids, attention_mask) 
                        st.write(f'--Коментар номер {i+1}--')
                        show_predict(LABELS, predictions, probs)

        
