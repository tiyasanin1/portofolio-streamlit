import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].apply(lambda x: iris.target_names[x])

# Streamlit app layout
st.title("Iris Flower Classification App")
st.write("""
### This app uses the Iris dataset to classify flowers into three species: 
- Setosa
- Versicolor
- Virginica
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])
input_df = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(input_df)

# Visualization: Pairplot
if st.checkbox("Show Pairplot of the Dataset"):
    st.subheader("Pairplot of the Iris Dataset")
    viz = sns.pairplot(iris_df, hue='species', height=2.5)
    st.pyplot(viz.figure)

# Split data into training and testing sets
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, pred)
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")


# Prediction based on user input
st.subheader("Prediction Based on User Input")
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write(f"Predicted Species: {iris.target_names[prediction][0]}")
st.write("Prediction Probability:")
st.write(prediction_proba)

# Plot Feature Importances
st.subheader("Feature Importances")
importance = pd.Series(model.feature_importances_, index=iris.feature_names)
viz_imp = importance.plot(kind='barh')
st.pyplot(viz_imp.figure)

