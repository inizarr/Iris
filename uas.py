import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Upload the dataset
st.title("D-Tree Web Application")
uploaded_file = st.file_uploader("Choose a file")

# Check if a file is uploaded
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(Iris.csv)

    # Check if the dataset has at least 2 columns
    if len(df.columns) >= 2:
        # Show the first 10 rows of the dataset
        st.write("First 10 rows of the dataset:")
        st.dataframe(df.head(10))

        # Choose the target column
        target_col = st.selectbox("Select the target column", df.columns)

        # Split the dataset into features and target
        features = df.drop(columns=[target_col])
        target = df[target_col]

        # Split the dataset into training and testing sets
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Create the Decision Tree classifier
        clf = DecisionTreeClassifier()

        # Train the classifier
        clf.fit(features_train, target_train)

        # Make predictions
        target_pred = clf.predict(features_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(target_test, target_pred)

        # Show the accuracy of the model
        st.write(f"Accuracy of the model: {accuracy:.2f}")

        # Create the plotly express tree plot
        tree_plot = px.tree(
            clf.fit(features_train, target_train),
            path=['True', 'False', 'color = LightSeaGreen'],
            values=features_train.iloc[:5, 1].values,
            hover_data=df.iloc[:5, 2:],
            title='Decision Tree Plot'
        )

        # Display the plotly express tree plot
        st.plotly_chart(tree_plot)
    else:
        st.write("Please upload a dataset with at least 2 columns.")
else:
    st.write("Please upload a dataset.")
