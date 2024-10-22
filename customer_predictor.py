import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def convert_strings_to_floats(data):
    # Convert all string columns to floats where possible
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column is of string type
            # Attempt to convert the column to numeric, coercing errors to NaN
            data[column] = pd.to_numeric(data[column], errors='coerce')
    return data


def predict_customer_behavior(data):
    # Preprocessing
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Convert string data to floats
    data_processed = convert_strings_to_floats(data_imputed)

    # Encode categorical variables (if any remain after conversion)
    le = LabelEncoder()
    for column in data_processed.columns:
        if data_processed[column].dtype == 'object':
            data_processed[column] = le.fit_transform(data_processed[column].astype(str))

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
