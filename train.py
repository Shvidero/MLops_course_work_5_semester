import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

mlflow.set_experiment("sentiments_training")

df = pd.read_csv('data/labeled_ds.csv', sep=';', encoding='utf-8')

# РАЗБИЕНИЕ НА TRAIN/TEST #

X = df['processed_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ОБУЧЕНИЕ МОДЕЛЕЙ И ИХ ЛОГГИРОВАНИЕ #

#   Эксперимент 1 Logistic Regression   #
with mlflow.start_run(run_name='LogReg_baseline'):
    
    mlflow.log_param("dataset_version", "v2")
    mlflow.log_param("model", "LogisticRegression")
    
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_features", 10000)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model_lr = LogisticRegression(
        max_iter=1000, 
        class_weight='balanced',
        random_state=42
    )
    model_lr.fit(X_train_vec, y_train)

    y_pred_lr = model_lr.predict(X_test_vec)
    
    print(f"Отчет о работе модели Logistic Regression\nTF_IDF: ngram_range=(1, 2), max_features=10000\n{classification_report(y_test, y_pred_lr)}")
        
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred_lr, average="macro"))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_lr))

    # Сохранение артефактов для FastApi и Docker
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model_lr, "artifacts/model.pkl")
    joblib.dump(vectorizer, "artifacts/vectorizer.pkl")

    mlflow.log_artifacts("artifacts")

    mlflow.log_artifact("data/labeled_ds.csv")
    
    
#   Эксперимент 2 Random Forest    #
with mlflow.start_run(run_name="RandomForest"):
    
    mlflow.log_param("dataset_version", "v2")
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print(f"Отчет о работе модели Random Forest\nTF_IDF: ngram_range=(1, 2), max_features=10000\n{classification_report(y_test, preds)}")

    mlflow.log_metric("f1_macro", f1_score(y_test, preds, average="macro"))
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    
    
#   Эксперимент 3 Logistic Regression №2    #
with mlflow.start_run(run_name='LogReg_baseline2'):
    
    mlflow.log_param("dataset_version", "v2")
    mlflow.log_param("model", "LogisticRegression")
    
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,3)")
    mlflow.log_param("max_features", 15000)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=15000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    
    print(f"Отчет о работе модели Logistic Regression\nTF_IDF: ngram_range=(1, 3), max_features=15000\n{classification_report(y_test, y_pred)}")
    
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    
