import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm


def preprocess_text(text):
    # 1. Приведение текста к нижнему регистру
    text = text.lower()

    # 2. Удаление ссылок
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # 3. Удаление хэштегов
    text = re.sub(r'#\w+', ' ', text)

    # 4. Удаление упоминаний
    text = re.sub(r'(id\d+|@\w+|club\d+)', ' ', text)

    # 5. Удаление всего, кроме букв, цифр, пробелов и ! ? .
    text = re.sub(r"[^a-zа-я0-9\s!?\.]", " ", text)
    
    # 6. Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()
        
    return text


# EDA #

df = pd.read_csv('Posts.csv', sep=';', encoding='utf-8')

df = df.drop(columns=['goal'])

# Заменяем пустые строки, состоящие из пробельных символов на Nan знаечния
df = df.replace(r'^\s*$', np.nan, regex=True)

# Удаляем строки с пустыми значениями и дубликаты
df = df.dropna()
df = df.drop_duplicates()


# Data preprocessing #

#### Виды упомнинаний:
# 1. ([id51469957|@katarinushka])
# 2. [id576145788|Игорь Коваленко] 
# 3. [club38981315|Российских железных дорог] и [club177946067|поезда "Сапсан"]

# Применение препроцессинга к датасету
if 'Text' in df.columns:
    tqdm.pandas()
    df['processed_text'] = df['Text'].progress_apply(preprocess_text)
    print("Предобработка текста завершена.")
    
# Доп обработка:

# Удаление текстов длиной короче 3 символов
df = df[df['processed_text'].str.split().str.len() >= 3]

# Обработка аномально длинных постов
df = df[df["processed_text"].str.split().str.len() < 500]


df.to_csv('preprocessed_ds.csv', sep=';', encoding='utf-8', index=False)