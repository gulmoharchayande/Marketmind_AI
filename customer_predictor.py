import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def predict_customer_behavior(data):
    # Preprocessing
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Encode categorical variables
    le = LabelEncoder()
    for column in data_imputed.columns:
        if data_imputed[column].dtype == 'object':
            data_imputed[column] = le.fit_transform(data_imputed[column].astype(str))

    # Assume the last column is the target variable
    X = data_imputed.iloc[:, :-1]
    y = data_imputed.iloc[:, -1]

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
