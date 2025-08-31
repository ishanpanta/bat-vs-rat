"""We write all function here but no actual input or outpu. Flake8 and pylint
is used to standarlize code with docstring , whitepasces
and comments"""
import pandas as pd


def read_data(path):
    """Read csv data with encoding"""
    return pd.read_csv(path, encoding="utf-8-sig")


def missing_values_find(df, name):
    """Check Missing Values"""
    print(f"\nMissing values in {name}:")
    print(df.isna().sum())