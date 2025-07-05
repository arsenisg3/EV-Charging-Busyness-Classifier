from data_fields import SessionFields as SF, TrainingFields as TF
from parameters import Parameters
import pandas as pd
import numpy as np


class InteractingFeatures:
    """Creates interactive features and the features necessary for their creation."""
    @staticmethod
    def create_shift(df, shifted_feature, feature_to_shift, group_cols=None, sort_cols=None) -> pd.DataFrame:
        """
        Create a shifted feature from specified features, grouped by charging point ID and sorted by date.

        Args:
            df (pd.DataFrame): The input DataFrame.
            shifted_feature (str): Name of the new shifted feature column.
            feature_to_shift (str): Name of the feature to shift.
            group_cols (list, optional): Columns to group by before shifting. Defaults to [SF.ID].
            sort_cols (list, optional): Columns to sort by within groups. Defaults to [TF.START_DATE].

        Returns:
            pd.DataFrame: DataFrame with the new shifted feature added.
        """
        if group_cols is None:
            if Parameters.AGGREGATE == 'points':
                group_cols = [SF.ID, TF.DAY_PERIOD]
            elif Parameters.AGGREGATE == 'stations':
                group_cols = [SF.LOCATION, TF.DAY_PERIOD]
        if sort_cols is None:
            sort_cols = [TF.START_DATE]

        # Sort the DF
        df = df.sort_values(group_cols + sort_cols).copy()

        if feature_to_shift == TF.HOLIDAY:
            df[shifted_feature] = df.groupby(group_cols)[feature_to_shift].transform(lambda x: np.roll(x, shift=1))
        else:
            df[shifted_feature] = df.groupby(group_cols)[feature_to_shift].transform(lambda x: x.shift(1))

        # Identify the indices where the shifted feature is NaN (first entries per group)
        nan_indices = df[shifted_feature].isna()
        if feature_to_shift == TF.HOLIDAY:
            df[shifted_feature] = df[shifted_feature].fillna(0)
        elif feature_to_shift == TF.WEEKDAY_ENCODED:
            # Calculate the previous day of the week for the NaN rows based on the current day's weekday
            current_day_weekday = df.loc[nan_indices, feature_to_shift]

            previous_weekday = (current_day_weekday - 1 + 7) % 7
            df.loc[nan_indices, shifted_feature] = previous_weekday
        else:
            df[shifted_feature] = df[shifted_feature].fillna(df[feature_to_shift].mean())

        return df

    @staticmethod
    def create_multiplication(df, new_feature, feature1, feature2) -> pd.DataFrame:
        """
        Create a feature resulting from the multiplication of two currently existing features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            new_feature (str): Name of the new feature column.
            feature1 (str): 1st feature required to create the new feature.
            feature2 (str): 2nd feature required to create the new feature.

        Returns:
            pd.DataFrame: DataFrame with the new feature added.
        """
        df[new_feature] = df[feature1] * df[feature2]
        return df

    @staticmethod
    def create_subtraction(df, new_feature, feature1, feature2) -> pd.DataFrame:
        """
        Create a feature resulting from the subtraction of two currently existing features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            new_feature (str): Name of the new feature column.
            feature1 (str): 1st feature required to create the new feature.
            feature2 (str): 2nd feature required to create the new feature.

        Returns:
            pd.DataFrame: DataFrame with the new feature added.
        """
        df[new_feature] = df[feature1] - df[feature2]
        return df


class FeaturesToDrop:
    """Drop features that are not of interest."""
    @staticmethod
    def features_to_drop(df, features_list) -> pd.DataFrame:
        for feature in features_list:
            df.drop(feature, axis=1, inplace=True)
        return df
