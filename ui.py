import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🚆 Sentiment Analysis of Railway Reviews")

st.write("Введите тексты (каждый с новой строки):")

texts = st.text_area("Texts", height=200)

if st.button("Predict"):
    if not texts.strip():
        st.warning("Введите хотя бы один текст")
    else:
        text_list = [t.strip() for t in texts.split("\n") if t.strip()]
        
        response = requests.post(
            "http://localhost:8000/predict",
            json={"texts": text_list}
        )

        if response.status_code == 200:
            results = response.json()
            for t, r in zip(text_list, results):
                st.markdown(
                    f"Текст: {t}\n\n"
                    f"➡️ Тональность: {r['sentiment']}\n\n"
                    f"Уверенность: {round(r['score'], 3)}\n\n---"
                )
        else:
            st.error("Ошибка при обращении к API")