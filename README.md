
# ETL Job and ML Pipeline Integration for Titanic Dataset

In this project, we have built an application on a real-world dataset sourced from Kaggle to perform the following tasks:
1. Data Extraction: Extracting data from a given source.
2. Data Transformation: Cleaning, preprocessing, and transforming the data to be suitable for ML models.
3. Data Loading: Loading the transformed data into a destination storage system or database.
4. Integration with ML Pipeline: Integrating the ETL job with a simple ML pipeline to demonstrate the end-to-end process.


## Table of Contents
1. Introduction
2. Objective
3. Dataset
4. Prerequisites
5. Setup
6. ETL Process
- Data Extraction
- Data Transformation
- Data Loading 
7. Integration with ML Pipeline
8. Model Evaluation
9. Database Schema Design
10. Conclusion
 


## Introduction
This project demonstrates the process of extracting, transforming, and loading (ETL) data in the context of a machine learning application. We use the Kaggle Titanic dataset to illustrate the end-to-end process from data extraction to machine learning model training.
## Dataset
The dataset used for this project is the Titanic dataset from Kaggle, which contains information about passengers aboard the Titanic. It is commonly used for introductory ML tasks.
## Prerequisites
- Python 3.7 or above
- Kaggle API installed and configured
- Libraries: pandas, sklearn, SQLAlchemy, sqlite3

## Setup
1. Download the Titanic dataset from Kaggle:
```shell
kaggle competitions download -c titanic
unzip titanic.zip
```
2. Install necessary python libraries
```shell
pip install pandas scikit-learn sqlalchemy
```


## ETL Process
### Data Extraction
Script:
```python

import pandas as pd

# Load the dataset
data = pd.read_csv('train.csv')

# Display the first few rows of the dataset
data.head()
```
Explanation:
We begin by downloading and extracting the Titanic dataset. Using the pandas library, we read the CSV file into a DataFrame to facilitate further processing.

### Data Transformation
Script:
```python
# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

# Perform feature engineering
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Normalize or scale numerical features and encode categorical features
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_cols = ['Age', 'Fare', 'FamilySize']
categorical_cols = ['Sex', 'Embarked', 'Title']

# Include 'Survived' in the original DataFrame for transformation
data = data[['Survived'] + numerical_cols + categorical_cols]

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

data_transformed = preprocessor.fit_transform(data.drop(columns=['Survived']))

transformed_columns = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
data_transformed_df = pd.DataFrame(data_transformed.toarray(), columns=transformed_columns)
data_transformed_df['Survived'] = data['Survived'].values

print(data_transformed_df.head())
```
Explanation:
- Handling Missing Values: We fill missing values in the 'Age' column with the median age, and the 'Embarked' column with the mode. We drop the 'Cabin' column due to excessive missing values.
- Feature Engineering: We extract titles from names and create a 'FamilySize' feature.
- Scaling and Encoding: We normalize numerical features using StandardScaler and encode categorical features using 'OneHotEncoder'.
### Data Loading
Script:
```python
from sqlalchemy import create_engine

# Create an SQLite database engine
engine = create_engine('sqlite:///titanic.db')

# Save the transformed DataFrame to a table named 'titanic_transformed'
data_transformed_df.to_sql('titanic_transformed', engine, index=False, if_exists='replace')

# Function to load data from the database
def load_data_from_db(engine):
    query = "SELECT * FROM titanic_transformed"
    data_from_db = pd.read_sql(query, engine)
    return data_from_db

data_loaded = load_data_from_db(engine)
print(data_loaded.head())
```
Explanation:
We save the cleaned and transformed data into an SQLite database. This demonstrates how to load data into a relational database using SQLAlchemy.


## Integration  with ML Pipeline
Script:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the features and target variable
X = data_loaded.drop(columns=['Survived'])
y = data_loaded['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```
Explanation:
We integrate the ETL job with a simple ML pipeline using scikit-learn. We train a logistic regression model and evaluate its performance.

## Model Evaluation
Result:
```python
Model Accuracy: 0.82
```
Explanation:
The logistic regression model achieved an accuracy of 82% on the test set. This demonstrates the complete process from data extraction to ML model training.

## Database Schema Design
The schema for the SQLite database created during the data loading process is implied by the 'to_sql' method. Here is the structure of the 'titanic_transformed' table:
```python
CREATE TABLE titanic_transformed (
    Age REAL,
    Fare REAL,
    FamilySize REAL,
    Sex_female INTEGER,
    Sex_male INTEGER,
    Embarked_C INTEGER,
    Embarked_Q INTEGER,
    Embarked_S INTEGER,
    Title_Capt INTEGER,
    Title_Col INTEGER,
    Title_Don INTEGER,
    Title_Dr INTEGER,
    Title_Jonkheer INTEGER,
    Title_Lady INTEGER,
    Title_Major INTEGER,
    Title_Master INTEGER,
    Title_Miss INTEGER,
    Title_Mlle INTEGER,
    Title_Mme INTEGER,
    Title_Mr INTEGER,
    Title_Mrs INTEGER,
    Title_Ms INTEGER,
    Title_Rev INTEGER,
    Title_Sir INTEGER,
    Title_the Countess INTEGER,
    Survived INTEGER
);
```
Explanation:
The table titanic_transformed includes normalized numerical features and one-hot encoded categorical features, along with the target variable Survived.
## Conclusion
This project illustrates the ETL process and its integration with a machine learning pipeline. We used the Kaggle Titanic dataset to demonstrate data extraction, transformation, and loading. We then built a simple logistic regression model to complete the end-to-end machine learning workflow.