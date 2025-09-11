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
    print(f"\nHandling missing data in: {name}")
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


def show_duplicate_data(df, name):
    """
    Detect and display duplicate rows in the dataset.
    """

    print(f"\nFinding duplicate data in: {name}")
    duplicates = df[df.duplicated(keep=False)]  # keep=False → mark all
    # duplicates, not just later ones
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate rows:\n")
        print(duplicates)
    else:
        print("No duplicates found")
    return duplicates


def handle_duplicates(df, name):
    """
    Detect and remove duplicate rows in the dataset.
    - First shows how many duplicates exist
    - Then removes them
    """
    print(f"\nHandling duplicate data in: {name}")
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


def clean_data_summary_till_this_stage(df, name="Dataset"):
    """
    Print summary of dataset (data types, missing values, duplicates).
    Does NOT modify or return the dataframe.
    """
    print(f"\nSummary for {name}:")
    df.info()  # no print() here, info() already prints
    print("\nMissing values per column:")
    print(df.isna().sum())

    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        print(f"\nFound {len(duplicates)} duplicate rows in {name}:")
        print(duplicates.to_string(index=False, line_width=2000))
    else:
        print(f"\nNo duplicates found in {name}.")


def convert_to_category(df, cols):
    """
    Convert given columns to categorical dtype.
    """
    for col in cols:
        if col in df.columns:
            df.loc[:, col] = df[col].astype("category")
            print(f"Converted {col} to category.")
    return df


def show_outlier_data(df, name="Dataset"):
    """
    Detect outliers in numeric columns using IQR.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"\nChecking outliers in {name}:")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} outliers detected.")
        else:
            print(f"{col}: No outliers detected.")
    print("Outlier check complete.\n")
    return df


def handle_outliers(df, name, columns=None, method="remove"):
    """
    Detect and handle outliers in numeric columns using the IQR method.
    Parameters:
        df (DataFrame): input data
        columns (list): list of numeric columns to check; if None, all numeric
        columns are used method (str): "remove" to drop rows with outliers,
        "cap" to cap values at bounds
    Returns:
        df_clean (DataFrame): dataframe with outliers handled
    """
    print(f"\nHandling outliers in {name}:")
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
        n_outliers = len(outliers)
        if n_outliers > 0:
            if method == "remove":  # Not going to be used in this data clean.
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col]
                                                                <= upper)]
                print(f"Outliers removed for {col}."
                      "New dataset has {len(df_clean)} rows.")
            elif method == "cap":
                df_clean[col] = df_clean[col].clip(lower, upper)
                print(f"Outliers capped for {col} at [{lower}, {upper}].")
        else:
            print(f"{col}: No outliers detected.")
    print("Outlier handling complete.\n")
    return df_clean


def clean_habit(value):
    if pd.isna(value):
        print("NaN  -->  unknown")
        return "unknown"

    text = str(value).strip().lower()

    if any(char.isdigit() for char in text):
        print(f"{value}  -->  invalid")
        return "invalid"

    replacements = {
        "rat_attack": "rat",
        "attack_rat": "rat",
        "rat attack": "rat",
        "bat_figiht": "bat_fight",
        "other_bat": "other",
        "other_bats": "other",
        "others": "other",
        "no food": "no_food",
        "pick_rat": "rat_pick",
        "rat_pick": "rat_pick",
    }

    if text in replacements:
        print(f"{value}  -->  {replacements[text]}")
        return replacements[text]

    # if no change, just return it
    return text
