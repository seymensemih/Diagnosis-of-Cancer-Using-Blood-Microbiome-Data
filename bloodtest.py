import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


data=pd.read_csv('data.csv')
label=pd.read_csv('labels.csv')
input=data.iloc[0:,1:]
target=label.iloc[0:,1]

for i in range(0,target.shape[0]):
    print(target[i])
for i in range(0,target.shape[0]):
    if target[i]== "colon cancer":
        target[i]=0
    elif target[i]== "lung cancer":
        target[i]=1
    elif target[i] == "breast cancer":
        target[i] = 2
    elif target[i]== "prostate cancer":
        target[i]=3

print(input.shape)
print(target.shape)

X=np.array(input).astype('float64')
y=np.array(target).astype('uint8')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regresyon = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
regresyon.fit(X_train, y_train)
y_pred = regresyon.predict(X_test)

XGBregressor = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
XGBregressor.fit(X_train,y_train)
xgb_pred=XGBregressor.predict(X_test)


print("\n\n\t\t\t\t     Random Forest\t\tGradient Boosting Trees")
print('Mean Absolute Error=    \t', metrics.mean_absolute_error(y_test, y_pred),"\t\t", metrics.mean_absolute_error(y_test, xgb_pred))
print('Mean Squared Error=     \t', metrics.mean_squared_error(y_test, y_pred),"\t\t", metrics.mean_squared_error(y_test, xgb_pred))
print('Root Mean Squared Error=\t', np.sqrt(metrics.mean_squared_error(y_test, y_pred)),"\t\t", np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))
print('Score=                  \t',regresyon.score(X_test,y_test),"\t\t", XGBregressor.score(X_test,y_test))

print("\n\n")

