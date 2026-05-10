# import pandas as pd
# import numpy as np
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # 1. Load the Wine dataset
# wine = load_wine()
# X = wine.data
# y = wine.target

# print(f"Dataset shape: {X.shape} (13 chemical features)\n")

# # 2. Train/Test Split (80% training, 20% testing)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- NO SCALING NEEDED! ---

# # 3. Initialize the Models
# # For the single tree, we use 'max_depth=3' to stop it from growing too deep and overfitting
# tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# # For the forest, 'n_estimators=100' means we are building a council of 100 separate trees
# forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # 4. Train the Models
# tree_model.fit(X_train, y_train)
# forest_model.fit(X_train, y_train)

# # 5. Predict and Evaluate
# tree_preds = tree_model.predict(X_test)
# forest_preds = forest_model.predict(X_test)

# print("--- Head-to-Head Results ---")
# print(f"Single Decision Tree Accuracy: {accuracy_score(y_test, tree_preds) * 100:.2f}%")
# print(f"Random Forest Accuracy:        {accuracy_score(y_test, forest_preds) * 100:.2f}%\n")


# # Extract the feature importances from the trained forest
# importances = forest_model.feature_importances_

# # Bind the feature names to their importance scores and sort them
# feature_names = wine.feature_names
# forest_importances = pd.Series(importances, index=feature_names)
# forest_importances = forest_importances.sort_values(ascending=False)

# print("--- Top 3 Most Important Features ---")
# print(forest_importances.head(3))


# import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

# # Create a large canvas to draw on
# plt.figure(figsize=(15, 10))

# # Draw the tree
# plot_tree(tree_model, 
#           feature_names=wine.feature_names, 
#           class_names=wine.target_names, 
#           filled=True,      # Colors the boxes based on the predicted class
#           rounded=True,     # Rounds the corners of the boxes
#           fontsize=10)

# # Display the drawing
# plt.show()


import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Load and Split the Data
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize the Models
# BAGGING: 100 independent trees voting at the end
bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)

# BOOSTING: 100 trees built in a chain, learning from the previous tree's mistakes
# 'learning_rate' controls how aggressively it tries to correct mistakes (0.1 is standard)
boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 3. Train the Models
bagging_model.fit(X_train, y_train)
boosting_model.fit(X_train, y_train)

# 4. Predict and Evaluate
bagging_preds = bagging_model.predict(X_test)
boosting_preds = boosting_model.predict(X_test)

print("--- The Ensemble Showdown ---")
print(f"Bagging (Random Forest) Accuracy:   {accuracy_score(y_test, bagging_preds) * 100:.2f}%")
print(f"Boosting (Gradient Boost) Accuracy: {accuracy_score(y_test, boosting_preds) * 100:.2f}%")