import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df=pd.read_csv("data/crop_yield.csv")
df.shape
data=df.head(20000)
print(data)

data.info()
data.describe()
data.isnull().sum()
data.duplicated().sum()

plt.subplots_adjust(hspace=0.5,wspace=0.5)
plt.subplot(2,2,1)
plt.bar(data["Crop"],data["Yield_tons_per_hectare"])
plt.xticks(rotation=90)
plt.xlabel("Crop")
plt.ylabel("Yield_tons_per_hectare")
plt.title("Crop Yield Visualization")

plt.subplot(2,2,2)
plt.scatter(data["Rainfall_mm"],data["Yield_tons_per_hectare"])
plt.xlabel("Rainfall_mm")
plt.ylabel("Yield_tons_per_hectare")
plt.title("Rainfall vs Yield")

plt.subplot(2,1,2)  
sns.boxplot(x='Soil_Type',y='Yield_tons_per_hectare',data=data)
plt.xlabel("Soil_Type")
plt.ylabel("Yield_tons_per_hectare")
plt.title("Soil Type vs Yield")

plt.tight_layout()
plt.show()

le_soil=LabelEncoder()
le_weather=LabelEncoder()
le_crop=LabelEncoder()
le_irrigation=LabelEncoder()
le_fertilizer=LabelEncoder()
data["Region"]=le_soil.fit_transform(data["Region"])
data["Soil_Type"]=le_soil.fit_transform(data["Soil_Type"])
data["Crop"]=le_crop.fit_transform(data["Crop"])
data["Weather_Condition"]=le_weather.fit_transform(data["Weather_Condition"])
data["Fertilizer_Used"]=le_fertilizer.fit_transform(data["Fertilizer_Used"])
data["Irrigation_Used"]=le_irrigation.fit_transform(data["Irrigation_Used"])
print("Classes for Soil_Type:", le_soil.classes_)
print("Classes for Crop:", le_crop.classes_)
print("Classes for Weather_Condition:", le_weather.classes_)
print("Classes for Fertilizer_Used:", le_fertilizer.classes_)
print("Classes for Irrigation_Used:", le_irrigation.classes_)
data.corr()

sns.heatmap(data.corr(),annot=True)
plt.show()

X=data.drop(["Yield_tons_per_hectare","Region","Days_to_Harvest"],axis=1)
y=data["Yield_tons_per_hectare"]
x=X.astype(int)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, input_dim=7, activation='relu'),  # Hidden layer 1
    Dense(32, activation='relu'),                # Hidden layer 2
    Dense(1, activation='linear')              # Output layer
])

model.compile(loss='mse', optimizer='adam', metrics=['mae','r2_score'])
history = model.fit(x_train_scaled, y_train, epochs=300, batch_size=400, validation_data=(x_test_scaled, y_test), verbose=1)

loss, r2_score, mae = model.evaluate(x_test_scaled, y_test)
print(f"MAE: {mae:.2f}%")
print(f"R2_score: {r2_score:.2f}%")

model.save("model/Crop_model.h5")
from tensorflow.keras.models import load_model
model = load_model("model/Crop_model.h5", compile=False)


compare=pd.DataFrame({'actual':y_test, 'predicted':model.predict(x_test_scaled).flatten()})
print(compare.head())

print("From the Given Situation")


Soil_Type=input("1.Type of the soil: ").strip().capitalize()
Crop=input("2.Which crop I can feed: ").strip().capitalize()
Rainfall_mm=int(input("3.Rainfall in mm: "))
Temperature_Celsius=int(input("4.Temperature: "))
Weather_Condition=input("5.Weather_Condition: ")
Fertilizer_Use=input("6.Fertilizer_Used: ").strip().capitalize()
Irrigation_Use=input("7.Irrigation_Used: ").strip().capitalize()

Fertilizer_Used = 'True' if Fertilizer_Use == 'Yes' else 'False'
Irrigation_Used = 'True' if Irrigation_Use == 'Yes' else 'False'


Soil_Type_enc=le_soil.transform([Soil_Type])[0]
Crop_encode=le_crop.transform([Crop])[0]
Weather_Condition_enc=le_weather.transform([Weather_Condition])[0]
Fertilizer_Used_enc=le_fertilizer.transform([Fertilizer_Use])[0]
Irrigation_Used_enc=le_irrigation.transform([Irrigation_Use])[0]

input_data=np.array([[Soil_Type_enc,Crop_encode,Rainfall_mm,Temperature_Celsius,Weather_Condition_enc,Fertilizer_Used_enc,Irrigation_Used_enc]])
scaledin_data=scaler.transform(input_data)
pred=model.predict(scaledin_data)[0]
print(f"\nYield Tons Per Hectare: {pred[0]:.3f} ton")
