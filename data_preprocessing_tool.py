import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path='Data.csv', test_size=0.2, random_state=1):
    """
    Preprocess a dataset by performing the following steps:
    1. Load the dataset from a CSV file.
    2. Handle missing values using mean imputation.
    3. Encode categorical features using one-hot encoding.
    4. Encode labels using label encoding.
    5. Split the dataset into training and testing sets.
    6. Standardize numerical features using StandardScaler.

    Parameters:
    - file_path (str): The path to the CSV file containing the dataset.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Seed for random number generation.

    Returns:
    - X_train (array): Features for training.
    - X_test (array): Features for testing.
    - y_train (array): Labels for training.
    - y_test (array): Labels for testing.
    """
    # Step 1: Load the dataset
    dataset = pd.read_csv(file_path)

    # Features (X) and labels (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Step 2: Handle missing values using mean imputation
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])

    # Step 3: Encode categorical features using one-hot encoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    # Step 4: Encode labels using label encoding
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Step 5: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 6: Standardize numerical features using StandardScaler
    sc = StandardScaler()
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])

    return X_train, X_test, y_train, y_test

# Example usage:
X_train, X_test, y_train, y_test = preprocess_data()
print("Training Features:")
print(X_train)
print("Testing Features:")
print(X_test)
print("Training Labels:")
print(y_train)
print("Testing Labels:")
print(y_test)
