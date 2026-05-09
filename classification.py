import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# print(iris_df.head())
print (iris_df.describe(include='all'))

X = iris_df.drop('target', axis=1)
y = iris_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

Knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='manhattan'
    
    )
Knn.fit(X_train_scaled, y_train)
predictions = Knn.predict(X_test_scaled)
print(predictions)
print(y_test.values)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
