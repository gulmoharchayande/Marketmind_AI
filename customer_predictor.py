import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def preprocess_data(data):
    # Replace specific non-numeric values with NaN
    data.replace({"N/A": None, "unknown": None, "": None}, inplace=True)
    return data


def convert_to_numeric(data):
    # Convert all columns to numeric where possible
    for column in data.columns:
        # Attempt to convert the column to numeric, coercing errors to NaN
        data[column] = pd.to_numeric(data[column], errors='coerce')
    return data


def encode_categorical(data):
    # Encode all remaining categorical variables
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column is of string type
            data[column] = le.fit_transform(data[column].astype(str))
    return data


def predict_customer_behavior(data):
    # Preprocessing
    # Step 1: Preprocess the data to handle specific non-numeric values
    data = preprocess_data(data)

    # Step 2: Convert all data to numeric values where possible
    data = convert_to_numeric(data)

    # Step 3: Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Ensure that the imputed DataFrame retains the same columns
    data_imputed.columns = data.columns  # Retain original column names

    # Step 4: Encode any remaining categorical variables
    data_processed = encode_categorical(data_imputed)

    # Assume the last column is the target variable
    X = data_processed.iloc[:, :-1]
    y = data_processed.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Create a DataFrame with the predictions
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)

    return results, accuracy
