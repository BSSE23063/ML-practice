import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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


# design the pipeline
Pipeline=Pipeline(
    [
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]
)

# train the model
Pipeline.fit(X_train,y_train)

predictions=Pipeline.predict(X_test)

print(predictions[:5].round(2))
print(y_test[:5].values.round(2))

mae=mean_absolute_error(y_test, predictions)
r2=r2_score(y_test, predictions)
   
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")