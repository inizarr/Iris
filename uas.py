import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

def load_data():
    # load dataset
    df = pd.read_csv('Iris.csv')

    x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df[['Species']]

    return df, x, y

def train_model(x,y):
    model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=80, splitter='best'
        )
    
    model.fit(x,y)

    score = model.score(x,y)

    return model, score

def predict(x,y, features):
    model, score = train_model(x,y)

    prediction = model.predict(np.array(features).reshape(1,-1))

    return prediction, score

def app():
    # Judul halaman aplikasi
    st.title("Aplikasi Pendataan Siswa Diterima Kerja")

    # Membuat sidebar
    st.sidebar.title("Navigasi")

    # Membuat radio option
    page = st.sidebar.radio("Pages", ["Home", "Prediction", "Visualisation"])

    # Load dataset
    df, x, y = load_data()

    # Kondisi call app function
    if page == "Prediction":
        app_prediction(df, x, y)
    elif page == "Visualisation":
        app_visualisation(df, x, y)
    else:
        app_home()

def app_home():
    st.write("Ini adalah halaman utama")

def app_prediction(df, x, y):
    st.write("Ini adalah halaman prediksi")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.number_input('Input nilai gender')
        ssc_p = st.number_input('Input nilai ssc_p')
        ssc_b = st.number_input('Input nilai ssc_b')
        hsc_p = st.number_input('Input nilai hsc_p')
        hsc_b = st.number_input('Input nilai hsc_b')
        hsc_s = st.number_input('Input nilai hsc_s')

    with col2:
        degree_p = st.number_input('Input nilai degree_p')
        degree_t = st.number_input('Input nilai degree_t')
        workex = st.number_input('Input nilai workex')
        etest_p = st.number_input('Input nilai etest_p')
        specialisation = st.number_input('Input nilai specialisation')
        mba_p = st.number_input('Input nilai mba_p')

    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    # Tombol prediksi
    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        score = score
        st.info("Prediksi Sukses...")

        if prediction == 1:
            st.warning("Siswa tidak ditempatkan")
        else:
            st.success("Siswa ditempatkan")

        st.write("Model yang digunakan memiliki tingkat akurasi", (score*100), "%")

def app_visualisation(df, x, y):
    st.write("Ini adalah halaman visualisasi")

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        dot_data = model.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['Siswa ditempatkan', 'Siswa yang tidak ditempatkan']
        )

        st.graphviz_chart(dot_data)

# Menjalankan aplikasi
app();
