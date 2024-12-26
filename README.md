# toxic_finder

# Опис проекту ...

# Інструкції по установці і запуску застусунку ...

# Команда:
* П.Коржов - скрам майстер
* Т.Дзвінчук - тім лід
* М.Соболь - розробник
* М.Шотурма - розробник
* О.Лозовий - розробник
* О.Сазонець - розробник

# Структура проекту:
* README.md - документація проекту. Виконавець Т.Дзвінчук.
* requirements.txt - перелік залежностей для проекту. Виконавець Т.Дзвінчук, П.Коржов.
* train_data.ipynb - попередня обробка тренувальних даних. Виконавець О.Лозовий. Фінальний результат зберегти у файл train_data.csv
* test_data.ipynb - попередня обробка тестових даних. Виконавець О.Лозовий. Фінальний результат зберегти у файл test_data.csv
* train_data.csv - оброблені тренувальні дані, готові до передачі у моделі для навчання.
* test_data.csv - оброблені тестові дані для фінальної оцінки та вибору найкращої моделі. 
* model_1.ipynb, model_2.ipynb, model_3.ipynb, model_4.ipynb - альтернативні версій створення і навчання моделі. Створені і навчені моделі зберігаємо у файли model_1.h5, model_2.h5, model_3.h5, model_4.h5 відповідно. Виконавці - М.Соболь, Т.Дзвінчук, М.Сазонець, М.Шотурма
* model_1.h5, model_2.h5, model_3.h5, model_4.h5 - навчені моделі
* estimate.ipynb - оценка чотирьох моделей, візуалізація. Вибір моделі, що демонструє найкращі результати та збереження її у файл best_model.h5. Виконавець О.Сазонець
* best_model.h5 - найкраща модель
* app.py - застосунок для оцінки коментарів за рівнем токсичності. Виконавець М. Шорутма
* Dockerfile - файл для створення Docker-образу. Містить усі інструкції для встановлення залежностей, копіювання коду та запуску програми. Виконавець П.Коржов.
* docker-compose.yml - файл для автоматизації розгортання проекту в середовищі Docker. Містить опис сервісів, мереж та томів, необхідних для роботи. Виконавець П.Коржов.


***

## Conda (Налаштування та Віртуальне середовище)

Щоб зробити проєкт відтворюваним і забезпечити зручне управління пакетами, у цьому проєкті використовується Conda як менеджер пакетів і середовища (бажано щоб на вашому диску `C` було не менше 20 гігабайт). Нижче наведені кроки для налаштування середовища:



1. **Встановіть Conda:**
Якщо Conda ще не встановлена, ви можете завантажити її з офіційного сайту [Anaconda](https://www.anaconda.com/products/individual):

    - Перейдіть за посиланням;
    - Впишіть ваш gmail;
    - Натисніть кнопку Submit;
    - Перейдіть на пошту що вказали;
    - Оберіть відповідний Anaconda Installer (якщо тут наприклад пише Python 3.12 або інший який у вас встановлений локально, не зважайте на це взагалі);
    - Запустіть завантажений `.exe` файл з правами адміністратора;
    - Під час встановлення ніякі галочки не тиснемо, все залишаємо як є;
    - Після завантаження запуститься ANACONDA.NAVIGATOR, але він нам не потрібен, його можна просто закрити;

    - Все, `conda` має бути додана в `PATH`, щоб це перевірити, відкрийте звичайну командну строку та виконайте наступну команду: `conda --version`
    - Якщо все добре, має вивистись версія `conda`, але якщо пише щось по типу що команду не розпізнано, значить `conda` не було додано в `PATH`, і вам треба зробити це вручну.


2. **Створіть нове середовище:** 
Після того як ви вже склонували репозиторій, відкрийте термінал у відповідній дерикторії з проектом і виконайте наступну команду, щоб створити нове середовище Conda з Python 3.9:

    ```bash
    conda create --name new_conda_env python=3.9
    ```


3. **Активуйте середовище:**
Після створення середовища активуйте його за допомогою команди:

    ```bash
    conda activate new_conda_env
    ```


4. **Встановіть необхідні пакети:**

    ```bash
    conda install jupyter numpy matplotlib seaborn pandas scikit-learn tensorflow keras transformers
    ```


5. **Запустіть Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```


Наступного разу під час роботи з проектом, буде достатньо виконати лище команди №3 та №5





















***


# Системні вимоги
Повний перелік залежностей поданий у файлі requirements.txt
# інструкціі для колабораторів
* роботу над своїми частинами проекту необхідно виконувати у власній гілці.
* перед початком роботи оновіть свою локальну копію main:

git checkout main
git pull origin main
* Встановіть або оновіть залежності:

pip install -r requirements.txt
* перейдіть на свою гілку:

git checkout -b ваша_гілка
* можете продовжувати роботу
* при додані бібліотек додавайте їх у файл requirements.txt

pip install назва_бібліотеки
pip freeze > requirements.txt
* створюйте Pull Request лише після тестування і забезпечення того, що ваш код працює з останніми залежностями
