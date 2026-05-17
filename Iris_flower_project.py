


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



iris = load_iris()

X = iris.data
y = iris.target


df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y

print("\nDataset Preview:")
print(df.head())



sns.pairplot(df, hue="species")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



print("\n--- Predict New Flower ---")


sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("Predicted Flower Type:",
      iris.target_names[prediction][0])



plt.figure()
plt.bar(["KNN Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.show()
