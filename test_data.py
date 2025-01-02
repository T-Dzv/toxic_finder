import pandas as pd
import re
import zipfile
import os
from transformers import BertTokenizer
import logging

# Зменшення рівня логування TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Шлях до ZIP-файлу
zip_file_path = os.path.join(os.path.dirname(__file__), 'jigsaw-toxic-comment-classification-challenge.zip')

# Розпаковка основного ZIP-файлу
extracted_path = os.path.join(os.path.dirname(__file__), 'extracted_data')
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Розпаковка вкладених ZIP-файлів для test.csv і test_labels.csv
def extract_inner_zip(inner_zip_name, destination):
    inner_zip_path = os.path.join(extracted_path, inner_zip_name)
    with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip_ref:
        inner_zip_ref.extractall(destination)

# Розпакування вкладених файлів
data_path = os.path.join(extracted_path, 'data')
os.makedirs(data_path, exist_ok=True)
extract_inner_zip('test.csv.zip', data_path)
extract_inner_zip('test_labels.csv.zip', data_path)

# Знаходження шляхів до test.csv і test_labels.csv
test_file_path = os.path.join(data_path, 'test.csv')
test_labels_file_path = os.path.join(data_path, 'test_labels.csv')

# Завантаження даних із test і test_labels
test_data = pd.read_csv(test_file_path)
test_labels = pd.read_csv(test_labels_file_path)

# Об'єднання даних із файлів за ID
merged_data = pd.merge(test_data, test_labels, on='id')

# Функція очищення тексту
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Замінює кілька пробілів одним
    text = re.sub(r'[^\w\s]', '', text)  # Видаляє все, крім букв, цифр і пробілів
    text = text.strip()  # Видаляє пробіли на початку і в кінці
    return text

# Застосування очищення тексту
merged_data['cleaned_comment_text'] = merged_data['comment_text'].apply(clean_text)

# Видалення рядків з мітками -1
filtered_data = merged_data[~((merged_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] == -1).any(axis=1))].copy()

# Ініціалізація токенізатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Функція для токенізації тексту для БЕРТ
def tokenize_text(text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf',  # TensorFlow замість PyTorch
        return_attention_mask=True
    )
<<<<<<< HEAD
    return encoding['input_ids'].numpy().squeeze(0).tolist(), encoding['attention_mask'].numpy().squeeze(0).tolist()
=======
    return encoding['input_ids'], encoding['attention_mask']
>>>>>>> f310650e5a2c1b990048111d70a92080bbd6079d

# Токенізація текстів із використанням .loc
input_ids, attention_masks = zip(*filtered_data['cleaned_comment_text'].apply(
    lambda x: tokenize_text(x, tokenizer)
))
filtered_data.loc[:, 'input_ids'] = input_ids
filtered_data.loc[:, 'attention_masks'] = attention_masks

# Збереження результату в CSV
<<<<<<< HEAD
output_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
filtered_data[['id', 'cleaned_comment_text', 'input_ids', 'attention_masks', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(output_path, index=False)

print(f"Обробка завершена. Результат збережено у {output_path}")
=======
output_path = os.path.join(os.path.dirname(__file__), 'test_data_cleaned.csv')
filtered_data[['id', 'cleaned_comment_text', 'input_ids', 'attention_masks', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(output_path, index=False)

print(f"Обробка завершена. Результат збережено у {output_path}")
>>>>>>> f310650e5a2c1b990048111d70a92080bbd6079d
