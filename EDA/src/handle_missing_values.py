import logging
from abc import ABC, abstractmethod
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract base class for missing values handling:
class MissingValuesHandlingStrat(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for handling

        :param df: Dataset we're working with
        :type df: pd.DataFrame
        :return: Dataframe with missing values handled.
        :rtype: DataFrame
        """
        pass


# Concrete Strategy for dropping missing values
class DropMissingvaluesStrat(MissingValuesHandlingStrat):
    def __init__(self, axis=0, thresh=None):
        """
        Initialize the class; drop missing values stategy with specific arguments.

        param axis: should drop rows with missing values, 1 to drop columns, 0 to drop rows
        :param thresh: the threshold for none-NA values.

        """
        self._axis = axis
        self._thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis. 1 for columns, 0 for rows.

        :param df: Dataset to consider.
        :type df: pd.DataFrame
        :return: Dataset without missing values.
        :rtype: DataFrame
        """

        logging.info(
            f"Dropping missing values with axis = {self._axis} and threshold = {self._thresh}"
        )
        df_cleaned = df.dropna(axis=self._axis)
        logging.info("Missing values dropped.")
        return df_cleaned


# Concrete Strategy for filling missing values
class FillMissingValuesStrat(MissingValuesHandlingStrat):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the fill missing values strategy class

        :param method: The method by which to fill in the missing values, could be the mean, median, mode or any value of your choice.
        :param fill_value: Description
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method

        :param df: The dataset we're modifying
        """
        logging.info(f"Filling missing values using method: {self}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(
                f"Unknown method '{self.method}'.No missing values handled."
            )

        logging.info("Missing values filled.")
        return df_cleaned


# concrete class for handling missing values
class MissingValueHandler:
    def __init__(self, strategy: MissingValuesHandlingStrat):
        """
        Initializes the missing value handler with specific parameters for the strategy selected.


        :param strategy: Strategy chosen
        :type strategy: MissingValuesHandlingStrat
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValuesHandlingStrat):
        """
        Sets a new strategy for the MissingvalueHandler.

        :param strategy: The new strategy we want to switch to
        :type strategy: MissingValuesHandlingStrat
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def execute_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the strategy selected.


        :param df: Input dataframe we're handling.
        :type df: pd.DataFrame
        :return: The dataframe with its missing values handled.
        :rtype: DataFrame
        """

        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)
