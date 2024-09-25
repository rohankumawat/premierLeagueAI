import pandas as pd
import os

# Function to load data
def load_data(file_path):
    """
    Load data from a CSV or other file formats.
    Args:
        file_path (str): Path to the data file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# General preprocessing
def preprocess_data(matches):
    """
    Perform general preprocessing steps.
    Args:
        matches (pd.DataFrame): Original data.
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Remove unwanted columns
    matches = matches.drop(["notes", "comp"], axis=1, errors='ignore')

    # Change the 'date' datatype from object to datetime
    matches["date"] = pd.to_datetime(matches["date"])

    # Create target column: 1 if team won, 0 otherwise
    matches["target"] = (matches["result"] == "W").astype("int")

    # Convert categorical columns to numerical codes
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes

    # Convert 'time' column to hour
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

    # Extract day of the week from the 'date' column
    matches["day_code"] = matches["date"].dt.dayofweek

    return matches

# Model-specific preprocessing for rolling averages
def rolling_averages(group, cols, new_cols):
    """
    Calculate rolling averages for team performance.
    Args:
        group (pd.DataFrame): Grouped team data.
        cols (list): Columns for which rolling averages are needed.
        new_cols (list): New column names for rolling averages.
    Returns:
        pd.DataFrame: Data with rolling averages.
    """
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def apply_rolling_averages(matches, cols):
    """
    Apply rolling averages to the entire dataset.
    Args:
        matches (pd.DataFrame): Preprocessed matches data.
        cols (list): Columns for which rolling averages are needed.
    Returns:
        pd.DataFrame: Data with rolling averages.
    """
    new_cols = [f"{col}_rolling" for col in cols]
    matches_rolling = matches.groupby("team", group_keys=False).apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.reset_index(drop=True)
    return matches_rolling, new_cols

# Function to save the preprocessed data for further usage
def save_preprocessed_data(data, file_path):
    """
    Save preprocessed data to a CSV file.
    Args:
        data (pd.DataFrame): Preprocessed data.
        file_path (str): Path to save the preprocessed data.
    """
    directory = os.path.dirname(file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")

    # Drop any columns with 'Unnamed' in the name (if exists)
    data_cleaned = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    try:
        # Save the cleaned data without the index
        data_cleaned.to_csv(file_path, index=False)
        print(f"Preprocessed data saved to {file_path}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

if __name__ == "__main__":
    # Define file paths
    raw_data_path = "data/raw/matches_2021_2024.csv"
    preprocessed_data_path = "data/preprocessed/preprocessed_data.csv"

    # Load raw data
    matches = load_data(raw_data_path)
    if matches is not None:
        # General preprocessing
        matches = preprocess_data(matches)

        # Apply rolling averages for selected columns
        cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
        matches_rolling, new_cols = apply_rolling_averages(matches, cols)

        # Save the preprocessed data
        save_preprocessed_data(matches_rolling, preprocessed_data_path)