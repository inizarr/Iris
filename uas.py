import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Mengunduh dataset
st.title('Titanic Decision Tree')
data = pd.read_csv('iris.csv')

# Memilih kolom yang ingin di tampilkan
columns = data.columns.tolist()
columns.remove('survived')
feature_selection = st.multiselect('Pilih Fitur', columns)

# Memasukkan fitur dan target ke dalam variabel baru
X = data[feature_selection]
y = data['survived']

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Menghitung akurasi model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write('Akurasi Model: ', accuracy)

# Menampilkan tree model
st.pyplot(clf.plot_tree(clf))
