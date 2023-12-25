import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Upload the dataset
st.title("Aplikasi Web Pohon Keputusan")
uploaded_file = st.file_uploader("Pilih file")

# Periksa apakah file diunggah
if uploaded_file is not None:
    # Muat dataset
    df = pd.read_csv(uploaded_file)

    # Periksa apakah dataset memiliki setidaknya 2 kolom
    if len(df.columns) >= 2:
        # Tampilkan 10 baris pertama dari dataset
        st.write("10 baris pertama dari dataset:")
        st.dataframe(df.head(10))

        # Pilih kolom target
        target_col = st.selectbox("Pilih kolom target", df.columns)

        # Pisahkan dataset menjadi fitur dan target
        features = df.drop(columns=[target_col])
        target_train = df[target_col]

        # Pisahkan dataset menjadi set pelatihan dan pengujian
        features_train, features_test, target_train, target_test = train_test_split(
            features, target_train, test_size=0.2, random_state=42
        )

        # Buat klasifikasi Pohon Keputusan
        clf = DecisionTreeClassifier()

        # Hitung akurasi model
        accuracy = accuracy_score(target_test)

        # Tampilkan akurasi model
        st.write(f"Akurasi model: {accuracy:.2f}")
