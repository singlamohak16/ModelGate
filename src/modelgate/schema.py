"""Explicit column roles for the approved IBM Telco CSV, not a general validator."""

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
SEGMENT_COLUMN = "Contract"
TARGET_MAPPING = {"No": 0, "Yes": 1}
NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_COLUMNS = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
FEATURE_COLUMNS = (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS)
REQUIRED_COLUMNS = (ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
