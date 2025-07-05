import pandas as pd
from data_fields import SessionFields as SF
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataLoader:
    """Class to load and merge datasets."""
    def __init__(self, location, filepath, filepath2):
        """
        Initializes DataLoader with file paths.

        Args:
            location (str): The directory containing the files.
            filepath (str): Filepath to the first dataset file.
            filepath2 (str): Filepath to  the second dataset file.
        """
        self.location = location
        self.filepath = f'{self.location}/{filepath}'
        self.filepath2 = f'{self.location}/{filepath2}'
        self.data = None
        self.data2 = None
        self.dataframe = None

    def load_data(self, time_cols=None):
        """
        Loads the datasets from the files and performs basic validation.

        Args:
            time_cols (list, optional): List of columns to parse as datetime.

        Returns:
            tuple: Two DataFrames containing the loaded datasets.
        """
        try:
            logger.info(f"Loading data from {self.filepath}...")
            self.data = pd.read_csv(self.filepath, sep=',', parse_dates=time_cols)
            logger.info(f"Data loaded successfully! Shape: {self.data.shape}")

            logger.info(f"Loading data from {self.filepath2}...")
            self.data2 = pd.read_csv(self.filepath2, sep=',', usecols=[SF.ID, SF.TYPE])
            logger.info(f"Data loaded successfully! Shape: {self.data2.shape}")

        except FileNotFoundError as e:
            logger.error(f"Error: File not found - {e}")
            raise

        except Exception as e:
            logger.error(f"Error: An unexpected issue occurred while loading data - {e}")
            raise

        return self.data, self.data2

    def merge_data(self):
        """
        Merges charger type information into the main dataset.

        Returns:
            DataFrame: The merged dataset.
        """
        print("Merging data...")
        charger_type = self.data2.groupby(SF.ID)[SF.TYPE].first()
        self.data[SF.TYPE] = self.data[SF.ID].map(charger_type)
        self.dataframe = self.data
        print(f"Merging complete! Shape: {self.dataframe.shape}")
        return self.dataframe
