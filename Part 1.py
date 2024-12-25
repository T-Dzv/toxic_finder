import pandas as pd
import re
from transformers import BertTokenizer

file_path = 'C:/Users/oleksii.lozovyi.LDLPROJECT/Downloads/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv'
data = pd.read_csv(file_path)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

data['cleaned_comment_text'] = data['comment_text'].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True
    )
    return encoding['input_ids'].squeeze(0).tolist(), encoding['attention_mask'].squeeze(0).tolist()

data['input_ids'], data['attention_masks'] = zip(*data['cleaned_comment_text'].apply(
    lambda x: tokenize_text(x, tokenizer)))

data[['cleaned_comment_text', 'input_ids', 'attention_masks']].to_csv('C:/Users/oleksii.lozovyi.LDLPROJECT/GoIT/toxic_finder/test_data.csv', index=False)