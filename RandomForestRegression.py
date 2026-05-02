import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

df=pd.read_csv('house_prices.csv')

# print(df.describe(include='all'))   

# print(df.head())
# print(df.info())
# print(df.shape)

# df.drop(df[df['price']<0].index, inplace=True)

# bad_price = df[df['price']<0]

# print(bad_price)
# print(len(bad_price))

# scalar = StandardScaler()

# load and clean the data

df=pd.read_csv('house_prices.csv')
df=df[df['price']>=0]

X=df.drop('price', axis=1)
y=df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Trying the Random Forest Regressor

# RRegression_pipeline=Pipeline([
#     ('scalar', StandardScaler()),
#     ('model',RandomForestRegressor(n_estimators=100,random_state=42))
# ])

# RRegression_pipeline.fit(X_train,y_train)

# RR_predictions=RRegression_pipeline.predict(X_test)

# rf_mae=mean_absolute_error(y_test,RR_predictions)
# rf_r2=r2_score(y_test,RR_predictions)


# print("--- Option 1: Random Forest ---")
# print(f"Mean Absolute Error: ${rf_mae:,.2f}")
# print(f"R-squared Score:     {rf_r2:.4f}")


poly_pipeline=Pipeline([
    ('scalar', StandardScaler()),
    
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge()),
])

# tring to find the best hyperparameters for Ridge regression using GridSearchCV

param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 200.0],
    
}

grid_search = GridSearchCV(poly_pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# poly_pipeline.fit(X_train, y_train)
poly_predictions = grid_search.predict(X_test)

best_model=grid_search.best_estimator_

tuned_predictions = best_model.predict(X_test)

tuned_mae = mean_absolute_error(y_test, tuned_predictions)
tuned_r2 = r2_score(y_test, tuned_predictions)
print(f"The Best Alpha value found: {grid_search.best_params_}")
print(f"Tuned MAE: ${tuned_mae:,.2f}")
print(f"Tuned R-squared: {tuned_r2:.4f}")


# poly_mae = mean_absolute_error(y_test, poly_predictions)
# poly_r2 = r2_score(y_test, poly_predictions)

# print("--- Option 2: Polynomial Regression ---")
# print(f"Mean Absolute Error: ${poly_mae:,.2f}")
# print(f"R-squared Score:     {poly_r2:.4f}")