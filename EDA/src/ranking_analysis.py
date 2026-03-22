from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Abstract class
# ---------------------------------------------------------------
# This class serves as a template for the other ranking analysis classes. It's a common interface.


class RankAnalyzer(ABC):
    @abstractmethod
    def execute_analysis(self, df: pd.DataFrame, **kwargs):
        """
        All concrete classes would implement this method, modifying it for their specific purpose.

        :param df: The dataset to analyze
        :type df: pd.Dataframe
        """

        pass


# Concrete class

# ---------------------------------------------------------------
#  This is an implementation of the template class for single grouping analysis.


class Single_Grouping_Ranking(RankAnalyzer):
    def execute_analysis(
        self,
        df: pd.DataFrame,
        feature1: str,
        feature2: str,
        title: str,
        xlabel: str,
        ylabel: str,
        file_path: str,
    ) -> None:
        """
        Rank the dataset using a single grouping

        :param df: The dataset to analyze
         :type df: pd.Dataframe
         :param feature1: This feature is used in grouping the dataset.
         :type feature1: str
         :param feature2: feature to sort by (Do note that, we could also make feature 2 a grouping feature.)
         :type feature2: str
         :param title: The custom title for the rank chart
         :type title: str
        """

        rank = df.groupby(feature1)[feature2].sum().nlargest(10).reset_index()

        rank["label"] = rank[feature1]

        sns.barplot(data=rank, x=feature2, y="label", palette="viridis")

        plt.title = title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_path)
        plt.close()


# Concrete class

# ---------------------------------------------------------------
#  This is an implementation of the template class for double grouping analysis.


class Double_grouping_Ranking(RankAnalyzer):
    def execute_analysis(
        self,
        df: pd.DataFrame,
        feature1: str,
        feature2: str,
        feature3: str,
        title: str,
        xlabel: str,
        ylabel: str,
        file_path: str,
    ) -> None:
        """
        Rank the dataset using double grouping

        :param df: The dataset to analyze
         :type df: pd.Dataframe
         :param feature1: This feature is used in grouping the dataset.
         :type feature1: str
         :param feature2: This feature is also used in grouping the dataset.
         :type feature2: str
         :param feature3: feature to sort by
         :type feature3: str
         :param title: The custom title for the rank chart
         :type title: str
         :param xlabel
         :param ylabel
         :filepath: Location to save ranking chart
        """

        rank = (
            df.groupby([feature1, feature2])[feature3].sum().nlargest(10).reset_index()
        )

        rank["label"] = rank[feature2] + "(" + rank[feature1] + ") "

        sns.barplot(data=rank, x=feature3, y="label", palette="viridis")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_path)


# Concrete class

# ---------------------------------------------------------------
#  This one helps for targetting specific data in the sorting column.


class Target_grouping_Ranking(RankAnalyzer):
    def execute_analysis(
        self,
        df: pd.DataFrame,
        target: str,
        feature1: str,
        feature2: str,
        feature3: str,
        title: str,
        xlabel: str,
        ylabel: str,
        filepath: str,
    ) -> None:
        """
        Rank the dataset using double grouping

        :param df: The dataset to analyze
         :type df: pd.Dataframe
         :param target: The target data we want to consider
         :param feature1: This feature is used in grouping the dataset.
         :type feature1: str
         :param feature2: This feature is also used in grouping the dataset.
         :type feature2: str
         :param feature3: feature to sort by
         :type feature3: str
         :param title: The custom title for the rank chart
         :type title: str
         :param xlabel
         :param ylabel
         :filepath: Location to save ranking chart
        """

        df_filtered = df[df[feature1] == target]

        rank = (
            df_filtered.groupby([feature1, feature2])[feature3]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        rank["label"] = rank[feature2]

        sns.barplot(data=rank, x=feature3, y="label", palette="viridis")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filepath)
        # plt.close()


class ranking_analyzer:
    def __init__(self, analyzer: RankAnalyzer):
        self._analyzer = analyzer

    def set_analyzer(self, analyzer: RankAnalyzer):
        self._analyzer = analyzer

    def run(self, df: pd.DataFrame, **kwargs) -> None:
        """Forwarding everything to the current strategy"""
        self._analyzer.execute_analysis(df, **kwargs)
