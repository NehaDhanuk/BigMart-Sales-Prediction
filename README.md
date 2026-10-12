# BigMart-Sales-Prediction
A machine learning project that predicts BigMart outlet sales using the XGBoost Regressor algorithm, with data preprocessing, exploratory data analysis, and model evaluation using Python.

BigMart Sales Prediction

About the Project

This project uses machine learning to predict sales of products across BigMart outlets. The XGBoost Regressor algorithm is used to train a model based on product and outlet features.

Objectives

- Perform Exploratory Data Analysis (EDA).
- Handle missing values in the dataset.
- Visualize product and outlet information.
- Encode categorical variables.
- Train a machine learning regression model.
- Evaluate model performance using R² score.
- Compare actual sales with predicted sales.

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

Project Workflow

1. Load and explore the dataset.
2. Clean the data and handle missing values.
3. Visualize data distributions.
4. Encode categorical features.
5. Split the dataset into training and testing sets.
6. Train the XGBoost Regressor.
7. Evaluate the model on training and testing data.
8. Save the trained model using Pickle.

How to Run the Project

1. Clone or download this repository.

2. Install the required libraries:
   
   "pip install -r requirements.txt"

3. Ensure the dataset is in the project folder.

4. Run the Python script:
   
   "python bigmart_sales_prediction.py"

Model Evaluation

The model's performance is evaluated using the R² score on the training and testing datasets.

Future Improvements

- Hyperparameter tuning.
- Improve categorical feature encoding.
- Evaluate using MAE and RMSE.
- Build a Streamlit web application for interactive sales prediction.
