🌾 Crop Yield Prediction Using Deep Learning

This project focuses on predicting crop yield in tons per hectare using environmental and agricultural factors such as soil type, crop type, rainfall, temperature, weather condition, fertilizer usage, and irrigation availability. Accurate crop yield prediction plays an important role in agriculture by helping farmers and planners make informed decisions. The project uses Python along with machine learning and deep learning libraries to analyze data, visualize patterns, train a predictive model, and generate yield predictions based on user inputs.

The dataset used in this project is a CSV file named crop_yield.csv, loaded from a local or Google Drive environment. The target variable is Yield_tons_per_hectare, while the input features include Soil_Type, Crop, Rainfall_mm, Temperature_Celsius, Weather_Condition, Fertilizer_Used, and Irrigation_Used. For efficient training and faster computation, only the first 20,000 rows of the dataset are used.

Several Python libraries are used in this project, including NumPy and Pandas for data handling, Matplotlib and Seaborn for data visualization, Scikit-learn for preprocessing tasks such as label encoding, feature scaling, and train-test splitting, and TensorFlow with Keras for building and training the deep learning model. The project is developed and tested using Google Colab.

Exploratory Data Analysis is performed to understand the dataset. This includes checking for missing values and duplicate records, generating descriptive statistics, and visualizing relationships between features and crop yield. Visualizations include a bar plot for crop versus yield, a scatter plot showing rainfall versus yield, a box plot comparing soil type and yield, and a correlation heatmap to identify relationships between numerical features.

Categorical variables such as soil type, crop, weather condition, fertilizer usage, and irrigation usage are converted into numerical form using LabelEncoder. Numerical features are standardized using StandardScaler to improve model performance. The dataset is then split into training and testing sets with an 80:20 ratio.

The prediction model is a deep learning regression model built using Keras. It consists of an input layer with seven features, two hidden dense layers with 64 and 32 neurons using ReLU activation, and an output layer with a single neuron using linear activation. The model is trained using the Adam optimizer with mean squared error as the loss function. Training is performed for 300 epochs with a batch size of 400. Model performance is evaluated using Mean Absolute Error and R² score.

After training, the model is saved as Crop_model.h5 and can be reloaded later for predictions without retraining. The model allows users to input custom agricultural and environmental conditions, including soil type, crop type, rainfall, temperature, weather condition, fertilizer usage, and irrigation usage. Based on these inputs, the model predicts the expected crop yield in tons per hectare.

This project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, visualization, model building, evaluation, and deployment-ready prediction logic. Future improvements may include using one-hot encoding for categorical variables, experimenting with other machine learning models such as Random Forest or XGBoost, performing hyperparameter tuning, and deploying the model as a web application using Flask or Streamlit.

Author: Yuvaraj A
