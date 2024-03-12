from typing import List, Tuple, Union

import pandas as pd


def compare_dataframes(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    index_a: str,
    index_b: str,
    cols_to_compare: Union[List, List[Tuple]],
    how: str,
) -> pd.DataFrame:
    """
    Compare two dataframes based on specified indices and columns.

    Args:
        df_a (pd.DataFrame): First dataframe to compare.
        df_b (pd.DataFrame): Second dataframe to compare.
        index_a (str): Name of the index column in df_a.
        index_b (str): Name of the index column in df_b.
        cols_to_compare (Union[List, List[Tuple]]): List of column names to compare
        or list of tuples specifying column names for each dataframe.
        how (str): Comparison method. Can be "inner", "left", or "right".

    Returns:
        pd.DataFrame: DataFrame containing comparison results as percentage of matches.
        pd.DataFrame: The merged dataframes used for comparison.
    """
    # Set indices if they are not already set
    if index_a != df_a.index.name:
        df_a = df_a.set_index(index_a)
    if index_b != df_b.index.name:
        df_b = df_b.set_index(index_b)

    if isinstance(cols_to_compare[0], tuple):
        cols_a, cols_b = map(list, zip(*cols_to_compare))
        print(cols_a)
    else:
        cols_a, cols_b = cols_to_compare, cols_to_compare

    # Ensure correct column names after merge operation
    cols_a_merged = [col + "_a" for col in cols_a]
    cols_b_merged = [col + "_b" for col in cols_b]

    comparison_results = df_a[cols_a].merge(
        df_b[cols_b], how=how, left_index=True, right_index=True, suffixes=("_a", "_b")
    )
    match_percentages = {}
    for col_a, col_b in zip(cols_a_merged, cols_b_merged):
        matches = (comparison_results[col_a] == comparison_results[col_b]).mean() * 100
        match_percentages[col_a[:-2]] = f"{matches:.2f}%"

    return (
        pd.DataFrame(match_percentages, index=["Match Percentage"]),
        comparison_results,
    )
