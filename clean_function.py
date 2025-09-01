"""We write all function here but no actual input or outpu. Flake8 and pylint
is used to standarlize code with docstring , whitepasces
and comments"""
import pandas as pd


def read_data(path):
    """Read csv data with encoding"""
    return pd.read_csv(path, encoding="utf-8-sig")


def check_few_values(df, name):
    print(f"\nChecking value for {name}:")
    print(df.head())


def missing_values_and_datatype_find(df, col):
    """Check Missing Values"""
    missing = df[col].isna().sum()
    print(f"Column: {col},", "dtype:", df[col].dtype, "missing:", missing,
          f"({missing/len(df):.1%})")


def detect_and_convert_datetime(df, name):
    """This checks the object datatype and converts it to date and 
    time if possible."""
    # Keep track of conversion results    
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
