# toxic_finder

***

# Опис проекту 

Додаток **toxic_finder** призначений для виявлення токсичних комментарів та визначення типу їх токсичності. 

Додаток може маркувати такі типи токсичності:
* 1 - Токсичний
* 2 - Сильно токсичний
* 3 - Непристойний
* 4 - Погрози
* 5 - Образи
* 6 - Ненависть до ідентичності
* 7 - Не токсичний

Користувач має можливість ввести (або завантажили із файлу) свій коментар та перевірити чи він є токсичним, та якого типу.

![image](https://github.com/user-attachments/assets/3da94246-ebcc-4745-94b1-2711b916df6b)

***Примітки:***
* Застосунок може обробляти лише коментарі, написані англійською мовою.
* Завантажений файл має містити не більше, ніж 5 коментарів.
* Застосунок може примати файли лише у форматі .txt, кожен коментар у файлі має почнатись із нового рядка
![image](https://github.com/user-attachments/assets/8d82b550-d922-44f6-abd5-ad99caeb1a26)

Для отримання результатів перевірки коментаря достатньо натиснути кнопку "Передбачити"

![image](https://github.com/user-attachments/assets/102ca062-2bc1-46b1-b5ef-47470b6f6e5b)


Застосунок **toxic_finder** має високу точність і майже безпомилково визначає типи токсичності у ваших коментарях. 

![image](https://github.com/user-attachments/assets/6677e2a8-a919-4536-a7e8-c2d0e2f59a6d)

***

# Інструкції по установці і запуску застусунку


### 1. **Клонування репозиторію**

```bash
git clone https://github.com/T-Dzv/toxic_finder.git
```

- **Результат:** Локальна копія репозиторію буде створена в папці `toxic_finder`.


### 2. **Перехід у папку репозиторію**

```bash
cd toxic_finder
```

### 3. **Запуск Docker Engine**

 - Для цього достатньо просто запустити програму Docker Desktop.

### 4. **Запуск контейнера з образом `toxic_finder`**

```bash
docker-compose up
```

Ця команда автоматично створить і запустить всі необхідні контейнери, описані у файлі `docker-compose.yaml`. Проєкт буде готовий до використання після завершення процесу запуску, для цього потрібно перейти за наступним посиланням: [http://localhost:8501](http://localhost:8501).

###### Про Docker Image:

Образ Docker було створено за допомогою `Dockerfile`, що включає всі необхідні інструкції для розгортання програми в контейнері. У процесі створення образу було виконано:
1. Вибір базового образу для контейнера.
2. Копіювання вихідного коду програми до контейнера.
3. Встановлення всіх необхідних залежностей.
4. Визначення команди для запуску програми в контейнері.
5. Завантаження готового образу на віддалений репозиторій DockerHub.

Цей підхід забезпечує простоту, зручність і універсальність використання проєкту в різних середовищах.

***

# Команда:
* П.Коржов - скрам майстер
* Т.Дзвінчук - тім лід
* М.Соболь - розробник
* М.Шотурма - розробник
* О.Лозовий - розробник
* О.Сазонець - розробник

***

# Структура проекту:
У зв'язку із обмеженнями GitHub робочі файли (датасети, моделі, історія навчання) збережені на Google диску і доступні за [посиланням](https://drive.google.com/drive/folders/1s6cB_58D4os_sLxeZcXGL5Fck8wOjAyG?usp=sharing).

Вихідний датасет - [Toxic Comment Classification Challeng](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
* .gitignore - список файлів та директорій, що не мають потрапити до репозиторію на GitHub.
* README.md - документація проекту. Виконавець Т.Дзвінчук.
* requirements.txt - перелік залежностей для проекту. Виконавець П.Коржов.
* train_data_1.py - попередня обробка тренувальних даних. Виконавець О.Лозовий. 
* test_data.py - попередня обробка тестових даних. Виконавець О.Лозовий. 
* train_data.csv, test_data.csv  - файли доступні на Google диску. Оброблені тренувальні та тестові дані, готові до передачі у моделі для навчання.
* model-1.ipynb, model-2.ipynb, model-3-new.ipynb, model-4-part2.ipynb, model-5.ipynb - альтернативні версій створення і навчання моделі. Виконавці - М.Соболь, Т.Дзвінчук
* model-1.h5, model_2.h5, model_3.h5, model_4_1_fin.h5, model_4_3_fin.h5, model-5.h5 - файли доступні на Google диску. Навчені моделі.
* training_history_model-1.json, history_4_1_fin.json, history_4_3_fin.json, training_history_5.json - файли доступні на Google диску. Збережена історія навчання моделей (не збережено для моделей №2 і №3)
* evaluation.ipynb - оценка чотирьох моделей, візуалізація. Вибір моделі, що демонструє найкращі результати та збереження її у файл best_model.h5. Виконавець О.Сазонець
* best_model.h5 - найкраща модель
* app.py - застосунок для оцінки коментарів за рівнем токсичності. Виконавець М. Шорутма
* Dockerfile - файл для створення Docker-образу. Містить усі інструкції для встановлення залежностей, копіювання коду та запуску програми. Виконавець П.Коржов.
* docker-compose.yaml - файл для автоматизації розгортання проекту в середовищі Docker. Виконавець П.Коржов.
* environment.yml - файл з усіма залежностями, який використовує conda для швидкого створення віртуального середовища. Виконавець П.Коржов.

***

# Опис підготовки даних

Виконавець модулю - О.Лозовий

Підготовка даних виконана у два етапи: підготовка тренувального датасету та підготовка тестового датасету.

Під час підготовки виконано:
* Завантаження вихідних даних та розпаковка заархівованих файлів
* Для тестових даних: об'єднання файлів зі зразками та з мітками за ID
* Очищення тексту: видалення зайвих пробілів, табуляції та переходів рядків, видалення символів, які не є літерами, цифрами або пробілами.
* Токенізація тексту за допомогою токенізатора BertTokenizer, підготовлюючи дані для моделі BERT.
* Збереження підготовленого датасету у форматі csv

***

# Опис моделей 

**Модель 1.** Виконавець - М.Соболь.
* Архітектура: Використовується попередньо натренована модель bert-base-uncased із бібліотеки Transformers без додаткових шарів, адаптована для задачі класифікації. Модель має вихідний шар для класифікації, кількість нейронів якого дорівнює кількості категорій у списку LABEL_COLUMNS (6 класів).
* Оптимізатор: Використовується оптимізатор Adam зі швидкістю навчання 5e-5, що підходить для тонкого налаштування попередньо натренованих моделей.
* Функція втрат: SparseCategoricalCrossentropy
* Метрика: Обчислюється точність (accuracy) для оцінки якості класифікації.
* Врахування незбалансованості даних: виконане балансування шліхом оверсемплінгу даних із менше ніж 1000 прикладами. 
* Навчання: Модель навчалась 3 епохи та продемонструвала гарний прогрес по покращенню цільових метрик.
* Модель має логічну помилку - при застосуванні np.argmax до міток у форматі one-hot вони були перетворені на одновимірні значення класу, що призвело до втрати багатоміткової природи даних. У результаті кожен зразок був позначений лише одним класом, замість того щоб зберігати інформацію про кілька активних міток одночасно.
* В результаті модель на тестових даних показує низьку точність.

**Модель 2.** Виконавець - Т.Дзвінчук.
* Архітектура: Модель побудована на кастомному шарі BERT із замороженими вагами, доповненому шарами GlobalAveragePooling1D, Dense (256 нейронів, активація swish), Dropout (0.3), та вихідним шаром із 6 нейронами й активацією sigmoid для багатоміткової класифікації.
* Оптимізатор: Використовується Nadam зі швидкістю навчання 1e-4 для плавного оновлення ваг.
* Функція втрат: Визначена кастомна функція втрат weighted_f1_loss, яка враховує ваги класів для оптимізації F1-метрики.
* Метрики: Оцінюються точність (accuracy), точність класифікації (precision), повнота (recall) та кастомна F1-метрика (f1_metric).
* Врахування незбалансованості даних: використані ваги класів, які враховані у кастомній функції втрат. 
* Навчання: використано ранню зупинку за метрикою val_f1_metric, навчання зупинено після першої епохи. Фінтюнинг моделі не проводився.
* Функція втрат була побудована на основі F1-метрики, яка відома своєю складністю для прямої оптимізації. Як наслідок, модель зіткнулася зі стагнацією під час навчання, що призвело до нездатності ефективно покращувати цільову метрику.
* В результаті модель на тестових даних показує низьку точність та демонтрує схильність до надмірного присвоєння класів токсичності нетоксичним прикладам. 

**Модель 3.** Виконавець - Т.Дзвінчук.
* Архітектура: Модель використовує кастомний шар BERT із замороженими вагами. Вихідний шар має 6 нейронів з активацією sigmoid для багатоміткової класифікації.
* Оптимізатор: Використовується Adam зі швидкістю навчання 5e-5, що підходить для тонкого налаштування попередньо натренованих моделей.
* Функція втрат: BinaryCrossentropy
* Метрики: Обчислюється точність (accuracy) для оцінки якості класифікації.
* Врахування незбалансованості даних: незбалансованість даних не врахована.
* Навчання: 10 епох навчання
* Через незбалансованість тренувальної (і валідаційної) вибірок модель показала оманливо високі показники точності, проте виявилась нездатною відрізняти класи токсичності і всі приклади відносила до нетоксичних.

**Модель 4.** Виконавець - Т.Дзвінчук.

Реалізовано багатозадачний підхід, передбачення виконуються у два етапи: бінарна модель класифікує коментарі на токсичні та нетоксичні; Багатоміткова модель визначає тип токсичності лише для токсичних коментарів.
* Архітектура: Бінарна модель - Модель побудована на кастомному шарі BERT із замороженими вагами, доповненому шарами GlobalAveragePooling1D, Dense (128 нейронів, активація swish), Dropout (0.3), та вихідним шаром із 1 нейроном й активацією sigmoid для бінарної класифікації. У якості багатоміткової моделі використана модель №2, описана вище.
* Оптимізатор: Бінарна модель - Adam, багатоміткова модель - Nadam.
* Функція втрат: Бінарна модель - BinaryCrossentropy, багатоміткова модель - кастомна функція втрат weighted_f1_loss, яка враховує ваги класів для оптимізації F1-метрики.
* Метрики: Бінарна модель - accuracy, багатоміткова модель - оцінюються точність (accuracy), точність класифікації (precision), повнота (recall) та кастомна F1-метрика (f1_metric).
* Врахування незбалансованості даних: Бінарна модель - виконане балансування даних за допомогою SMOTE, багатоміткова модель - використані ваги класів, які враховані у кастомній функції втрат.
* Навчання: Бінарна модель - 5 епох навчання із замороженими шарами БЕРТ, та 3 епохи навчання під час фінтюнингу. Багатоміткова модель - виконаний фінтюнинг на 3 епохи.
* Модель має схильність до надмірного присвоєння класів токсичності та в цілому має досить низькі показники точності.

**Модель 5.** Виконавець - М.Соболь.
* Архітектура: Використовується попередньо натренована модель bert-base-uncased із бібліотеки Transformers без додаткових шарів, адаптована для задачі класифікації. Модель має вихідний шар для класифікації, кількість нейронів якого дорівнює кількості категорій у списку LABEL_COLUMNS (7 класів, додано неявний клас нетоксичних коментарів).
* Оптимізатор: Використовується оптимізатор Adam зі швидкістю навчання 5e-5, що підходить для тонкого налаштування попередньо натренованих моделей.
* Функція втрат: BinaryCrossentropy
* Метрика: Обчислюється точність (accuracy) для оцінки якості класифікації.
* Врахування незбалансованості даних: виконане балансування шляхом оверсемплінгу даних із менше ніж 1000 прикладами. 
* Навчання: Модель навчалась 3 епохи та продемонструвала гарний прогрес по покращенню цільових метрик.
* Модель схожа на модель №1, але із виправленням логічної помилки із передачою даних на навчання.
* Модель продемонструваа високі показники на оцінці, як на валідаційних, та і на тестових даних.
* ***Поточна модель вибрана у якості робочої для застосунку***

***

# Опис оцінки моделей

Виконавець модулю - О.Сазонець.

В рамках модулю виконана оцінка всіх п'яти моделей на тестових даних:
* Класифікаційний звіт: Precision, Recall, F1-score для кожного класу окремо та в цілому.
* Обчислення метрики F1 Score (мікро, середнє по всім прикладам)
* Обчислення метрики ROC-AUC для кожного класу із візуалізацією результатів
* Confusion Matrix для кожного класу із візуалізацією результатів
* Візуалізація історії навчання моделей для відстежування перенавчання
* Оцінка прогнозів на конкретних прикладах (порівняння прогнозів із фактичними мітками на декількох реальних прикладах із вибірки)

За результатами оцінки найкращою визнана модель №5:
* Класифікаційний звіт: Модель працює добре для найбільш поширеного класу нетоксичних коментарів, а також поширених категорій toxic, obscene, insult. Для рідких класів (severe_toxic, threat, identity_hate) якість передбачень нижче. Проте для настільки незбалансованої вибірки, результати передбачень досить високі.
* Метрика F1 Score по всім прикладам складає 0.8872056, що є дужа високим показником.
* ROC-AUC > 0.96 для всіх класів показує, що модель здатна ефективно розрізняти токсичність та нетоксичність, навіть для рідкісних класів, що є чудовим показником для мультікласової задачі класифікації.
* Confusion Matrix для кожного класу підтверджують висновки класифікаційного звіту: Модель працює добре для найбільш  поширених категорій toxic, obscene, insult, для рідких класів (severe_toxic, threat, identity_hate) якість передбачень нижче.
* Візуалізація історії навчання: модель демонструє певні ознаки перенавчання після 2-ї епохи (із трьох).
* Оцінка на конкретних прикладах: модель помилилась із визначенням severe_toxic на одному прикладі, але в цілому прогнози моделі близькі до істиних міток

***

# Системні вимоги

- Python — 3.9.18
- Jupyter — 1.0.0
- NumPy — 1.26.4
- Matplotlib — 3.9.2
- Seaborn — 0.13.2
- Pandas — 2.2.3
- Scikit-learn — 1.5.2
- Tensorflow — 2.11.0
- Keras — 2.11.0
- Transformers — 4.32.1
- Streamlit — 1.13.0
- Altair — 4.x.x



***

# Покрокові Git команди для роботи з репозиторієм [toxic_finder](https://github.com/T-Dzv/toxic_finder.git)

Цей посібник допоможе вам покроково зрозуміти, як працювати з репозиторієм на GitHub. Почнемо з клонування репозиторію і завершимо створенням гілки та її завантаженням на віддалений репозиторій.


### 1. **Клонування репозиторію**
Це дозволить скопіювати репозиторій `toxic_finder` на ваш комп'ютер.

```bash
git clone https://github.com/T-Dzv/toxic_finder.git
```

- **Результат:** Локальна копія репозиторію буде створена в папці `toxic_finder`.


### 2. **Перехід у папку репозиторію**
Перейдіть у створену папку, щоб працювати з репозиторієм.

```bash
cd toxic_finder
```

- **Результат:** Ви тепер працюєте всередині репозиторію.


### 3. **Перевірка статусу репозиторію**
Перевірте поточний стан репозиторію (які файли змінені, чи є щось для коміту).

```bash
git status
```

- **Результат:** Ви побачите, чи є зміни або нові файли, які потрібно додати.


### 4. **Перейдіть на гілку `main`(якщо ви зараз на іншій гілці)**

```bash
git checkout main
```

### 5. **Створення нової гілки**
Щоб працювати над окремим завданням чи функцією, створіть нову гілку.

```bash
git branch new_feature
```

- **`new_feature`:** Назва гілки (можете змінити на іншу, відповідно до вашої задачі).


### 6. **Перехід у нову гілку**
Після створення гілки перейдіть у неї.

```bash
git checkout new_feature
```

Або, починаючи з Git 2.23+, можна створити і перейти в нову гілку одночасно:

```bash
git switch -c new_feature
```

- **Результат:** Ви тепер працюєте в новій гілці.


### 7. **Внесення змін у проект**
- Редагуйте, додавайте або видаляйте файли в проєкті.
- Після внесення змін виконайте наступні кроки, щоб зберегти їх у Git.


### 8. **Додавання файлів до індексу**
Додайте змінені або нові файли до індексу (staging area).

```bash
git add .
```

- **`git add .`:** Додає всі змінені файли.
- **Альтернативно:** Можна додати конкретний файл:
  ```bash
  git add filename
  ```


### 9. **Створення коміту**
Збережіть ваші зміни в історії гілки.

```bash
git commit -m "Додано нову функцію"
```

- **`-m "Додано нову функцію"`:** Короткий опис змін.


### 10. **Перевірка гілок**
Перевірте, в якій гілці ви зараз перебуваєте, та перелік усіх доступних гілок.

```bash
git branch
```

- **Результат:** Поточна гілка буде позначена зірочкою `*`.


### 11. **Завантаження гілки на віддалений репозиторій**
Щоб завантажити нову гілку на GitHub, скористайтеся командою:

```bash
git push origin new_feature
```

- **`origin`:** Це стандартна назва віддаленого репозиторію.
- **`new_feature`:** Назва вашої гілки.


### 12. **Створення Pull Request (PR) на GitHub**
1. Відкрийте репозиторій на GitHub: [toxic_finder](https://github.com/T-Dzv/toxic_finder.git).
2. Перейдіть у вкладку **Pull Requests**.
3. Натисніть **New Pull Request**.
4. Виберіть вашу гілку `new_feature` для злиття в основну гілку (зазвичай `main`).
5. Додайте опис змін і натисніть **Create Pull Request**.

***

## Додаткові корисні команди

#### Відправити зміни до існуючої гілки
Якщо вже створили гілку на GitHub і хочете лише оновити її:

```bash
git push
```

#### Команда, що оновить ваш локальний репозиторій останніми змінами з `main` із віддаленого репозиторію
```bash
git pull origin main
```

#### **Скасувати останній коміт, але зберегти зміни в індексі (staging area):**
Якщо ви хочете скасувати коміт і залишити файли готовими для повторного коміту:

```bash
git reset --soft HEAD~1
```

- **`HEAD~1`** означає "останній коміт".
- **Результат:** зміни залишаться в індексі (`staging area`), і ви зможете зробити новий коміт.


#### **Скасувати останній коміт і повернути зміни в робочу директорію:**
Ця команда знімає зміни з індексу і повертає їх до робочої директорії:

```bash
git reset --mixed HEAD~1
```

- **Результат:** зміни повернуться до робочої директорії, але не будуть у staging.



#### **Скасувати останній коміт і видалити всі зміни (небезпечний варіант):**
Це скасовує останній коміт і повністю видаляє всі внесені зміни:

```bash
git reset --hard HEAD~1
```

- **Увага:** Використовуйте цю команду обережно, оскільки зміни буде втрачено назавжди.



#### **Скасувати певний коміт у середині історії:**
Якщо вам потрібно скасувати конкретний коміт (наприклад, серед попередніх), використовуйте:

```bash
git revert <commit_hash>
```

- Замість `<commit_hash>` вкажіть хеш коміту, який потрібно скасувати (знайдіть його через `git log`).
- **Результат:** Git створить новий коміт, який скасує зміни з обраного коміту.


#### **Скасувати останній коміт, якщо він уже завантажений на віддалений репозиторій:**
Якщо ви вже зробили `git push`, то:

1. Скасуйте коміт локально (будь-який з варіантів вище).
2. Завантажте зміни на сервер із примусовим перезаписом:

   ```bash
   git push origin branch_name --force
   ```

   (Замість `branch_name` вкажіть вашу гілку, наприклад, `main` або `feature_branch`.)


#### **Переглянути історію комітів:**
Щоб знайти потрібний коміт для скасування, скористайтеся:

```bash
git log
```

- Для короткого формату:

  ```bash
  git log --oneline
  ```

***

# Conda (Налаштування та Віртуальне середовище)

Щоб зробити проєкт відтворюваним і забезпечити зручне управління пакетами, у цьому проєкті використовується Conda як менеджер пакетів і середовища. Нижче наведені кроки для налаштування середовища:



1. **Встановіть Conda:** Якщо Conda ще не встановлена, ви можете завантажити її з офіційного сайту [Anaconda](https://www.anaconda.com/products/individual)


2. **Створіть нове середовище:** Після того як ви вже склонували репозиторій, відкрийте термінал у відповідній дерикторії з проектом і виконайте наступну команду, щоб створити нове середовище Conda з усіма необхідними залежностями:

    ```bash
    conda env create --file environment.yml
    ```


3. **Активуйте середовище:** Після створення середовища активуйте його за допомогою команди:

    ```bash
    conda activate teamproject
    ```


4. **Запустіть Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```


Наступного разу під час роботи з проектом, буде достатньо виконати лише останні дві команди
