"""We write all function here but no actual input or outpu. Flake8 and pylint
is used to standarlize code with docstring , whitepasces
and comments"""
import pandas as pd


def read_data(path):
    """Read csv data with encoding"""
    return pd.read_csv(path, encoding="utf-8-sig")


def check_few_values(df, name="Dataset"):
    print(f"\nChecking value for {name}:")
    print(df.head())


def missing_values_and_datatype_find(df, col):
    """Check Missing Values"""
    missing = df[col].isna().sum()
    print(f"Column: {col},", "dtype:", df[col].dtype, "missing:", missing,
          f"({missing/len(df):.1%})")


def detect_and_convert_datetime(df, name="Dataset"):
    """This checks the object datatype and converts it to date and
    time if possible."""
    # Keep track of conversion results.
    result = {}
    for col in df.columns:
        # Only check for object datatypes.
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M",
                                         errors="raise")
                # print(f"Converted column '{col}' to datetime")
                result[col] = "Converted to datetime"
            except Exception:
                # If conversion fails, leave it as object.
                # and record the result.
                result[col] = "Not a datetime column."

    # Print conversion result.
    print(f"\nDatetime conversion result for {name}")
    for col, status in result.items():
        print(f"{col}:{status}")
    return df


def detect_and_convert_number(df, name="Dataset"):
    """This checks the object datatype and converts it to number
    if possible."""
    # Keep track of conversion results.
    result = {}
    for col in df.columns:
        # Only check for object datatypes.
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
                # print(f"Converted column '{col}' to numberic datatype")
                result[col] = "Converted to numeric datatype."
            except Exception:
                # If conversion fails, leave it as object.
                # and record the result.
                result[col] = "Not a numeric column."

    # Print conversion result.
    print(f"\nNumeric conversion result for {name}")
    for col, status in result.items():
        print(f"{col}:{status}")
    return df


def datatype_conversion_summary(df, name="Dataset"):
    """Displays the summary of column types after conversionS"""
    print(f"\n Summary of {name}:")
    # Counts total of each datatype.
    print(df.dtypes.value_counts())
    print("\nDetailed View:")
    print(df.dtypes)


def handle_missing_values(df, name="Dataset"):
    """
    Handle missing values column by column.
    - Numeric: fill with median
    - Categorical (object): fill with mode
    """
    print(f"\nHandling dublicate data in: {name}")
    missing_data_count = 0
    for col in df.columns:
        if df[col].isna().sum() > 0:  # only if missing
            missing_data_count = missing_data_count + 1
            if df[col].dtype in ["int64", "float64"]:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"Missing in {col} filled with median = {median_value}")
            else:
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print(f"Missing in {col} filled with mode = {mode_value}")
    if missing_data_count < 1:
        print(f"No missing data in {name}") 
    return df


def detect_and_handle_duplicate_data(df):
    """
    Detect and remove duplicate rows in the dataset.
    - First shows how many duplicates exist
    - Then removes them
    """
    # Count duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Found {duplicate_count} duplicate rows")

    if duplicate_count > 0:
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"Removed {duplicate_count} duplicates")
    else:
        print("No duplicates found")
    return df
