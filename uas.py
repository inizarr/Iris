import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def build_model(features_train, target_train):
    clf = DecisionTreeClassifier()
    clf.fit(features_train, target_train)
    return clf

def plot_tree(clf, features_train, target_train):
    tree_plot = px.tree(
        clf.fit(features_train, target_train),
        path=['True', 'False', 'color = LightSeaGreen'],
        values=features_train.iloc[:5, 1].values,
        hover_data=df.iloc[:5, 2:],
        title='Decision Tree Plot'
    )
    return tree_plot

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

        # Bagi dataset menjadi fitur dan target
        features = df.drop(columns=[target_col])
        target = df[target_col]

        # Bagi dataset menjadi set pelatihan dan pengujian
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Bangun klasifikasi Pohon Keputusan
        clf = build_model(features_train, target_train)

        # Buat prediksi
        target_pred = clf.predict(features_test)

        # Hitung akurasi model
        accuracy = accuracy_score(target_test, target_pred)

        # Tampilkan akurasi model
        st.write(f"Akurasi model: {accuracy:.2f}")

        # Buat plot pohon menggunakan plotly express
        tree_plot = plot_tree(clf, features_train, target_train)

        # Tampilkan plot pohon menggunakan plotly express
        st.plotly_chart(tree_plot)
    else:
        st.write("Silakan unggah dataset dengan setidaknya 2 kolom.")
else:
    st.write("Silakan unggah dataset.")
