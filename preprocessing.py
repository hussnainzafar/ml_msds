import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path):
    """
    Load a dataset from a CSV file, split it into features (X) and labels (y),
    and then further split it into training and testing sets.

    Parameters:
    - file_path (str): The path to the CSV file containing the dataset.

    Returns:
    - X_train (array): Features for training.
    - X_test (array): Features for testing.
    - y_train (array): Labels for training.
    - y_test (array): Labels for testing.
    """
    # Load dataset from CSV file
    dataset = pd.read_csv(file_path)

    # Extract features (X) and labels (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    import pdb; pdb.set_trace()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

# Example usage:
file_path = './Data.csv'
X_train, X_test, y_train, y_test = load_and_split_data(file_path)

# Display the results
print("Training Features:")
print(X_train)
print("\nTesting Features:")
print(X_test)
print("\nTraining Labels:")
print(y_train)
print("\nTesting Labels:")
print(y_test)
