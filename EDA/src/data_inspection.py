from abc import ABC, abstractmethod

import pandas as pd


class DataInspection(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Point is to perform a specific inspection on the dataframe.

        Parameters:
        df(pd.Dataframe): The dataframe that we're inspecting

        Returns:
        Nothing, We just print out the inspection results.
        """

        print("Data types and Non-null counts:")
        print(df.info())
        pass


class DatatypesInspection(DataInspection):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the data in the csv

        Parameters:
        df(pd.Dataframe): The dataframe that we're inspecting

        Returns:
        Nothing, just prints the outputs.
        """

        print("\nData types and non-null counts")
        print(df.info())


class SummaryStatistics(DataInspection):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the categorical data in the csv

        Parameters:
        df(pd.Dataframe): The dataframe that we're inspecting

        Returns:
        Nothing, just prints the outputs.
        """

        print("\nSummary statistics (Numerical features):")
        print(df.describe())
        print("\nData shape: ", df.shape)
        print("\nSummary statistics (Categorical features):")
        print(df.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy: DataInspection):
        self._strategy = strategy
        print(self._strategy)

    def set_strategy(self, strategy: DataInspection):
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        self._strategy.inspect(df)


if __name__ == "__main__":
    data_path = "../File Ingestion/extracted_data/train.csv"
    df = pd.read_csv(data_path)

    # Initialize the inspector to inspect the datatypes.
    inspector = DataInspector(DatatypesInspection())
    inspector.execute_inspection(df)

    # Switch to get the summary statistics
    inspector.set_strategy(SummaryStatistics())
    inspector.execute_inspection(df)
