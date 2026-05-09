import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


cancer =load_breast_cancer()

x=cancer.data
y= cancer.target

scalar = StandardScaler()

x_scaled = scalar.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

svm.fit(x_train, y_train)

predictions = svm.predict(x_test)

print(predictions)
print(y_test)
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
