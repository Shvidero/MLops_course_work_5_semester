import mlflow
from mlflow.data import from_pandas
import pandas as pd

mlflow.set_experiment("sentiment_data__pipeline")

# 1. первая версия датасета
df_preprocessed = pd.read_csv("data/preprocessed_ds.csv", sep=';', encoding='utf-8')
with mlflow.start_run(run_name="dataset_v1"):

    ds_v1 = from_pandas(
        df_preprocessed,
        source="Posts.csv -> preprocessing",
        name="preprocessed_dataset"
    )

    mlflow.log_input(ds_v1, context="training")
    mlflow.log_param("dataset_version", "v1")


# 2. вторая версия датасета
df_labeled = pd.read_csv("data/labeled_ds.csv", sep=';', encoding='utf-8')
with mlflow.start_run(run_name="dataset_v2"):

    ds_v2 = from_pandas(
        df_labeled,
        source="preprocessed_ds.csv -> labeling",
        name="labeled_dataset"
    )

    mlflow.log_input(ds_v2, context="training")