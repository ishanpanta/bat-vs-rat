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


def missing_values_find(df, col):
    """Check Missing Values"""
    print(f"\n--- Column: {col} ---")
    print("dtype:", df[col].dtype)
    missing = df[col].isna().sum()
    print("missing:", missing, f"({missing/len(df):.1%})")
