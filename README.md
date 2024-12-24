# toxic_finder

# Опис проекту ...

# Інструкції по установці і запуску застусунку ...

# Структура проекту:
* README.md - документація проекту. Виконавець Т.Дзвінчук.
* requirements.txt - перелік залежностей для проекту. Виконавець Т.Дзвінчук, П.Коржов.
* data.ipynb - попередня обробка даних. Виконавець О.Лозовий. Фінальний результат дберегти у файл data.csv
* data.csv - оброблені дані, готові до передачі у проект.
* model_1.ipynb, model_2.ipynb, model_3.ipynb - альтернативні версій створення і навчання моделі. Створені і навчені можелі зберігаємо у файли model_1.h5, model_2.h5, model_3.h5 відповідно. Виконавці - М.Соболь, Т.Дзвітнчук та ...
* model_1.h5, model_2.h5, model_3.h5 - навчені моделі
* estimate.ipynb - оценка трьох моделей, візуалізація. Вибір моделі, що демонструє найкращі результати та збереження її у файл best_model.h5. Виконавець О.Сазонець
* best_model.h5 - найкраща модель
* app.py - застосунок для оцінки коментарів за рывнем токсичності. Виконавець М. Шорутма
* Dockerfile - файл для створення Docker-образу. Містить усі інструкції для встановлення залежностей, копіювання коду та запуску програми. Виконавець П.Коржов.
* docker-compose.yml - файл для автоматизації розгортання проекту в середовищі Docker. Містить опис сервісів, мереж та томів, необхідних для роботи. Виконавець П.Коржов.

# Системні вимоги
Повний перелік залежностей поданий у файлі requirements.txt
* # інструкціі для колабораторів
* перед початком роботи оновіть свою локальну копію main:

git checkout main
git pull origin main
* Встановіть або оновіть залежності:

pip install -r requirements.txt
* перейдіть на свою гілку:
* 
