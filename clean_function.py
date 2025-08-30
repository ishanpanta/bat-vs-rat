"""We write all function here but no actual input or outpu. Flake8 and pylint
is used to standarlize code with docstring , whitepasces
and comments"""
import pandas as pd


def read_data(path):
    """Read csv data with encoding"""
    return pd.read_csv(path, encoding="utf-8-sig")


def common_issues_find(df, col):
    """Find common issues in the dataset"""
    missing = df[col].isna().sum()
    print("missing:", missing, f"({missing/len(df):.1%})")
    # Check for non-numerical data.
    if df[col].dtype == object:
        # show top unique values (to spot noise)
        print("sample unique values (up to 20):")
        print(pd.Series(df[col].astype(str).unique()[:20]))
        # detect numeric-like strings that will become NaN if coerced
        # pattern:
        # # ^-? → optional minus sign
        # # \d+ → digits
        # # (\.\d+)? → optional decimal
        # ~ → invert → selects non-numeric values.
        # df[col].notna() → ignore missing values.
        non_numeric = df[~df[col].astype(str).str.match(r"^-?\d+(\.\d+)?$") & df[col].notna()]
        if len(non_numeric) > 0:
            print(
                    "Non pure-numeric examples "
                    "(for object -> numeric coercion):\n",
                    non_numeric.head(5).to_dict(orient='records'))
    else:
        # numeric or datetime
        print(df[col].describe(include='all'))
