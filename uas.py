import streamlit as st
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Memuat dataset iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Decision Tree
model = DecisionTreeClassifier()

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Memprediksi kelas untuk data uji
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan hasil prediksi dan akurasi
st.write("Hasil Prediksi:", y_pred)
st.write("Akurasi:", accuracy)
