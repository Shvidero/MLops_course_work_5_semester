import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Для разметки данных была выбрана модель RuBERT для тональности:

# blanchefort/rubert-base-cased-sentiment

# Причины:
# 1. обучена именно на русскоязычных отзывах и сообщениях
# 2. поддерживает 3 класса: positive / neutral / negative
# 3. совместима с HuggingFace

df = pd.read_csv('data/preprocessed_ds.csv', sep=';', encoding='utf-8')

# Формирования пайплайна
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment",
    tokenizer="blanchefort/rubert-base-cased-sentiment"
)

texts = df["processed_text"].astype(str).tolist()

results = []
batch_size = 16

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    preds = sentiment_model(batch)
    results.extend(preds)

df["sentiment"] = [r["label"] for r in results]
df["sentiment_score"] = [r["score"] for r in results]

# Сохранение в csv
df.to_csv('data/labeled_ds.csv', sep=';', encoding='utf-8', index=False)