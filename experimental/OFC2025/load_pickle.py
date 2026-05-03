import pickle
import pandas as pd


def load_pickled_dataframes(file_path):
    """
    Load a pickled dictionary of DataFrames from a file.

    Args:
    file_path (str): Path to the pickled file

    Returns:
    dict: Dictionary of pandas DataFrames
    """
    try:
        with open(file_path, 'rb') as file:
            data_dict = pickle.load(file)

        # Verify that the loaded object is a dictionary of DataFrames
        if not isinstance(data_dict, dict):
            raise ValueError("Loaded object is not a dictionary")

        for key, value in data_dict.items():
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"Item with key '{key}' is not a DataFrame")

        print(f"Successfully loaded {len(data_dict)} DataFrames from {file_path}")
        return data_dict

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        print(f"Error: File at {file_path} is not a valid pickle file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return None


# Usage example
if __name__ == "__main__":
    file_path = "path/to/your/pickled_dataframes.pkl"
    loaded_data = load_pickled_dataframes(file_path)

    if loaded_data is not None:
        # Access individual DataFrames
        for key, df in loaded_data.items():
            print(f"\nDataFrame '{key}':")
            print(df.head())  # Print the first few rows of each DataFrame
            print(f"Shape: {df.shape}")
