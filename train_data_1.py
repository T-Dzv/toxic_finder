import pandas as pd
import re
import zipfile
import os
from transformers import BertTokenizer



# Шлях до ZIP-файлу
zip_file_path = os.path.join(os.path.dirname(__file__), 'jigsaw-toxic-comment-classification-challenge.zip')

# Розпаковка основного ZIP-файлу
extracted_path = os.path.join(os.path.dirname(__file__), 'extracted_data')
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Розпакування вкладених файлів
data_path = os.path.join(extracted_path, 'data')
os.makedirs(data_path, exist_ok=True)
inner_zip_path = os.path.join(extracted_path, 'train.csv.zip')
with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip_ref:
    inner_zip_ref.extractall(data_path)

# Знаходження шляху до train.csv
train_file_path = os.path.join(data_path, 'train.csv')

# Завантаження даних із train
data = pd.read_csv(train_file_path)

# Функція очищення тексту
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# Застосування очищення тексту
data['cleaned_comment_text'] = data['comment_text'].apply(clean_text)

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

# Токенізація текстів
data['input_ids'], data['attention_masks'] = zip(*data['cleaned_comment_text'].apply(
    lambda x: tokenize_text(x, tokenizer)))

# Збереження результату в CSV
output_path = os.path.join(os.path.dirname(__file__), 'train_data.csv')
data[['id', 'cleaned_comment_text', 'input_ids', 'attention_masks', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(output_path, index=False)

print(f"Обробка завершена. Результат збережено у {output_path}")
