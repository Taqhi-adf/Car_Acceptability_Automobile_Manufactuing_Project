#•	unacc/acc → bad (0)

#💻 3. Python Implementation (Using Random Forest)
#Here’s the full working example #👇
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load dataset (UCI dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=columns)

# Display first few rows
print(df.head())

# Simplify target: good/vgood → good (1), acc/unacc → bad (0)
df['class'] = df['class'].replace({'unacc': 0, 'acc': 0, 'good': 1, 'vgood': 1})
print(df['class'])

# Encode categorical features
le = LabelEncoder()
for col in ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']:
    df[col] = le.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)


# Predict
y_pred1 = model1.predict(X_test)

# Evaluate model
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"\nAccuracy: {accuracy1:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("\nClassification Report:\n", classification_report(y_test, y_pred1))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred1), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Car Acceptability Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Show feature importance
feature_importance = pd.Series(model1.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=True).plot(kind='barh', color='teal')
plt.title("Feature Importance in Car Acceptability (Random Forest)")
plt.show()


# Initialize Random Forest Classifier
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, y_train)

# Predict
y_pred2 = model2.predict(X_test)

# Evaluate model
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"\nAccuracy: {accuracy2:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("\nClassification Report:\n", classification_report(y_test, y_pred2))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Car Acceptability Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Show feature importance
feature_importance = pd.Series(model2.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=True).plot(kind='barh', color='teal')
plt.title("Feature Importance in Car Acceptability (Random Forest)")
plt.show()


# Initialize Random Forest Classifier
model3 = MultinomialNB()
model3.fit(X_train, y_train)

# Predict
y_pred3 = model3.predict(X_test)

# Evaluate model
accuracy3 = accuracy_score(y_test, y_pred3)
print(f"\nAccuracy: {accuracy3:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("\nClassification Report:\n", classification_report(y_test, y_pred3))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred3), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Car Acceptability Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Show feature importance
#feature_importance = pd.Series(model3.feature_importances_, index=X.columns)
#feature_importance.sort_values(ascending=True).plot(kind='barh', color='teal')
#plt.title("Feature Importance in Car Acceptability (Random Forest)")
#plt.show()


model1.results = pd.DataFrame(["RandomForestClassifier",y_pred1,accuracy1,confusion_matrix]).transpose()
model1.results.columns = ["Model","Predictions","Accuracy","Confusion Matrix"]
print(model1.results)

model2.results = pd.DataFrame(["DecisionTreeClassifier",y_pred2,accuracy2,confusion_matrix]).transpose()
model2.results.columns = ["Model","Predictions","Accuracy","Confusion Matrix"]
print(model2.results)

model3.results = pd.DataFrame(["MultinomialNB",y_pred3,accuracy3,confusion_matrix]).transpose()
model3.results.columns = ["Model","Predictions","Accuracy","Confusion Matrix"]
print(model3.results)

df_models = pd.concat([model1.results, model2.results, model3.results],axis=0)
df_models.reset_index()

# the end of the code is to compare the results of all the models in a single dataframe for better visualization and comparison.
