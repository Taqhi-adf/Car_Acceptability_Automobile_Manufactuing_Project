import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder,_encoders,OneHotEncoder
import matplotlib.pyplot as plt

# -------------------------------
# Title
# -------------------------------
st.title("🚗 Car Evaluation ML App")
st.write("Predict Car Acceptability (Good / Bad) using Multiple ML Models")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(url, names=columns)
    return df

data = load_data()

st.subheader("📊 Dataset Preview")
st.write(data.head())

# -------------------------------
# Preprocessing
# -------------------------------
data['class'] = data['class'].replace({
    'unacc': 0, 'acc': 0,
    'good': 1, 'vgood': 1
})

le = LabelEncoder()
encoders = {}

for col in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop('class', axis=1)
y = data['class']

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Models
# -------------------------------
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model2 = DecisionTreeClassifier(random_state=42)
model3 = MultinomialNB()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

acc1 = accuracy_score(y_test, y_pred1)
acc2 = accuracy_score(y_test, y_pred2)
acc3 = accuracy_score(y_test, y_pred3)

# -------------------------------
# Model Comparison
# -------------------------------
st.subheader("📈 Model Comparison")

results = pd.DataFrame({
    "Model": ["Random Forest", "Decision Tree", "Naive Bayes"],
    "Accuracy": [acc1, acc2, acc3]
})

st.dataframe(results)

# -------------------------------
# Plot Accuracy
# -------------------------------
fig, ax = plt.subplots()
ax.bar(results["Model"], results["Accuracy"])
ax.set_title("Model Accuracy Comparison")
ax.set_ylabel("Accuracy")
st.pyplot(fig)

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("📊 Confusion Matrix")

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "Decision Tree", "Naive Bayes"]
)

if model_choice == "Random Forest":
    cm = confusion_matrix(y_test, y_pred1)
elif model_choice == "Decision Tree":
    cm = confusion_matrix(y_test, y_pred2)
else:
    cm = confusion_matrix(y_test, y_pred3)

fig2, ax2 = plt.subplots()
ax2.imshow(cm)
ax2.set_title("Confusion Matrix")
st.pyplot(fig2)

# -------------------------------
# User Input Prediction
# -------------------------------
st.subheader("🔮 Predict Car Acceptability")

buying = st.selectbox("Buying Price", ['vhigh', 'high', 'med', 'low'])
maint = st.selectbox("Maintenance Cost", ['vhigh', 'high', 'med', 'low'])
doors = st.selectbox("Doors", ['2', '3', '4', '5more'])
persons = st.selectbox("Persons", ['2', '4', 'more'])
lug_boot = st.selectbox("Luggage Boot", ['small', 'med', 'big'])
safety = st.selectbox("Safety", ['low', 'med', 'high'])

# Encode input manually
def encode_input(val, col):
    return encoders[col].transform([val])[0]

input_data = np.array([[
    encode_input(buying, 'buying'),
    encode_input(maint, 'maint'),
    encode_input(doors, 'doors'),
    encode_input(persons, 'persons'),
    encode_input(lug_boot, 'lug_boot'),
    encode_input(safety, 'safety')
]])

# Prediction
model_select = st.selectbox(
    "Choose Model for Prediction",
    ["Random Forest", "Decision Tree", "Naive Bayes"]
)

if st.button("Predict"):
    if model_select == "Random Forest":
        pred = model1.predict(input_data)
    elif model_select == "Decision Tree":
        pred = model2.predict(input_data)
    else:
        pred = model3.predict(input_data)

    result = "✅ GOOD CAR" if pred[0] == 1 else "❌ BAD CAR"
    st.success(f"Prediction: {result}")

# -------------------------------
# Feature Importance (RF)
# -------------------------------
st.subheader("🔥 Feature Importance (Random Forest)")

importance = pd.Series(model1.feature_importances_, index=X.columns)
fig3, ax3 = plt.subplots()
importance.sort_values().plot(kind='barh', ax=ax3)
st.pyplot(fig3)