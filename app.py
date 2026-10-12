import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


data = pd.read_csv("Big_martSales.csv")
print(data.head())

print(data.info())
# print(data.isnull().sum())

print(data['Item_Weight'].mean())
data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
print(data.isnull().sum())

outlet_size_mode = data.pivot_table(values="Outlet_Size", columns="Outlet_Type", aggfunc=(lambda x : x.mode()))
print(outlet_size_mode)

# data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

missing_values_loc = data["Outlet_Size"].isnull()
print(missing_values_loc)

data.loc[missing_values_loc, "Outlet_Size"] = data.loc[missing_values_loc, "Outlet_Type"].apply(lambda x : outlet_size_mode[x])
print(data.isnull().sum())

print(data.describe())

#Data Distribution for Item_Weight Column
plt.figure(figsize=(6,6))
sns.displot(data["Item_Weight"])
# plt.show()

#Data Distribution for Item_Visibility Column
plt.figure(figsize=(6,6))
sns.displot(data["Item_Visibility"])
# plt.show()

#Data Distribution for Item_MRP Column
plt.figure(figsize=(6,6))
sns.displot(data["Item_MRP"])
# plt.show()

#Data Distribution for Item_Outlet_Sales Column
plt.figure(figsize=(6,6))
sns.displot(data["Item_Outlet_Sales"])
# plt.show()

#Count plot for Outlet_Establishment_Year Column
plt.figure(figsize=(6,6))
sns.countplot(x = data["Outlet_Establishment_Year"], data = data)
# plt.show()

#Count plot for Item_Fat_Content  Column
plt.figure(figsize=(6,6))
sns.countplot(x = data["Item_Fat_Content"], data = data)
# plt.show()

#Count plot for Item_Type  Column
plt.figure(figsize=(25,6))
sns.countplot(x = data["Item_Type"], data = data)
# plt.show()

#Count plot for Outlet_Size  Column
plt.figure(figsize=(6,6))
sns.countplot(x = data["Outlet_Size"], data = data)
# plt.show()

#Count plot for Outlet_Type  Column
plt.figure(figsize=(6,6))
sns.countplot(x = data["Outlet_Type"], data = data)
# plt.show()

print(data["Item_Fat_Content"].value_counts())
data.replace({"Item_Fat_Content" : {"LF" : "Low Fat", "low fat" : "Low Fat", "reg" : "Regular"}}, inplace=True)
print(data["Item_Fat_Content"].value_counts())

print(data.head())

encoder = LabelEncoder()
data["Item_Identifier"] = encoder.fit_transform(data["Item_Identifier"])
data["Item_Fat_Content"] = encoder.fit_transform(data["Item_Fat_Content"])
data["Item_Type"] = encoder.fit_transform(data["Item_Type"])
data["Outlet_Identifier"] = encoder.fit_transform(data["Outlet_Identifier"])
data["Outlet_Size"] = encoder.fit_transform(data["Outlet_Size"])
data["Outlet_Location_Type"] = encoder.fit_transform(data["Outlet_Location_Type"])
data["Outlet_Type"] = encoder.fit_transform(data["Outlet_Type"])

print(data["Item_Fat_Content"].value_counts())
# print(data.head())

x = data.drop(columns="Item_Outlet_Sales")
y = data["Item_Outlet_Sales"]

#Spliting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x.shape, x_train.shape, x_test.shape) 
print(y.shape, y_train.shape, y_test.shape) 

#Model Creation



#Model trai, y_traning

model = XGBRegressor()

model.fit(x_train, y_train)

from sklearn.metrics import r2_score

predictions_train = model.predict(x_train)

print("Train R2 Score: ",r2_score(y_train, predictions_train))

predictions_test = model.predict(x_test)

print("Test R2 Score: ",r2_score(y_test, predictions_test))


comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions_test
})

print(comparison.head(10))

#model
import pickle

pickle.dump(model, open("bigmart_model.pkl", "wb"))
