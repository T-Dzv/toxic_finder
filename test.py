import pandas as pd
import os


def validate_train_data(file_path):
    """
    Функція для перевірки коректності підготовлених даних для моделі BERT.
    :param file_path: str, шлях до файлу train_data.csv
    :return: None (друкує результати перевірки)
    """
    try:
        # Завантаження даних
        data = pd.read_csv(file_path)

        print("\nПеревірка структури файлу train_data.csv")
        print("--------------------------------------------------")

        # Перевірка, що файл завантажено коректно
        if data.empty:
            raise ValueError("Файл train_data.csv порожній.")

        # Перевірка обов'язкових колонок
        required_columns = ["cleaned_comment_text", "input_ids", "attention_masks"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Відсутні обов'язкові колонки: {', '.join(missing_columns)}")

        print("✔ Усі обов'язкові колонки присутні.")

        # Перевірка розміру даних
        num_rows = len(data)
        print(f"✔ Файл містить {num_rows} рядків.")

        if num_rows < 100:
            print("⚠ Попередження: Мала кількість рядків. Можливо, дані неповні.")

        # Перевірка коректності формату input_ids і attention_masks
        for col in ["input_ids", "attention_masks"]:
            if not data[col].apply(lambda x: isinstance(x, str)).all():
                raise ValueError(f"Колонка {col} містить некоректні дані. Очікувався строковий формат JSON.")

        print("✔ Формат колонок input_ids і attention_masks коректний.")

        print("\nФайл train_data.csv успішно пройшов усі перевірки!")

    except Exception as e:
        print(f"Помилка перевірки даних: {e}")

# Шлях до файлу train_data.csv
file_path = os.path.join(os.path.dirname(__file__), 'train_data.csv')

# Виклик функції перевірки
validate_train_data(file_path)