"""
Dataset creation methods for specific test cases, e.g. memory consumption.

Many machine learning libraries provide methods already to create datasets to
quickly run some experiments,
e.g. sklearn.datasets.make_classification()
(https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification).

This package provides additional dataset creation methods for specific test
cases, e.g. very large datasets to try out Cobra's memory consumption.
"""
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def make_large_house_prices_dataset(
        data_folder: str = './data/argentina-venta-de-propiedades',
        ask_download_confirmation: bool = True) -> pd.DataFrame:
    """
    Create a very large house prices dataset (shape: (2126816, 340))
    for classification and regression purposes, based on the Kaggle dataset at
    https://www.kaggle.com/msorondo/argentina-venta-de-propiedades.

    This method downloads the dataset from Kaggle,
    since including it directly in this repository would make the repository
    very large, while only a few users of this repository will use this dataset.

    To make the download work, you need to have a Kaggle API Token saved
    in a file on your computer.
    If this is not the case, a warning will be thrown to help you set this up.

    To execute this, we advise that your PC should have >= 16 GB RAM.

    Parameters
    ----------
    data_folder : str
        path of the folder in which the CSV files can be written
    ask_download_confirmation : bool
        whether a confirmation must be asked before the 2.47 GB of CSV files
        are downloaded. Set to False to run this method from a pytest unit test,
        because those don't suppport input() calls.

    Returns
    -------
    pd.DataFrame
        a 2Mx340 basetable, ready for classification or regression experiments.

    Raises
    ------
    ModuleNotFoundError
        In case the kaggle package is not installed.
        This is a dependency solely for this method, so it is not included
        in cobra's requirements.txt.
    IOError
        In case a Kaggle API token file is not available on your machine.
        See our help message printed in this case for how to solve this.
    """
    # Importing the following modules at the top of the file is not useful,
    # since they are only required for THIS specific dataset creation method.
    # We don't want to make other dataset creation methods crash on
    # the unavailability of the following modules if they don't use them.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from zipfile import ZipFile

    setup_help_msg = r"""
    This method downloads the dataset from Kaggle, 
    since including it directly in this repository would make the repository 
    very large, while only a few users of this repository will use this dataset.
    
    To make the download work, you need to have a Kaggle API Token saved 
    in a file on your computer.
    If this is not yet the case:
    1. Create a Kaggle account, if you don't have one yet.
    2. Log in on Kaggle's website.
    3. On your Kaggle account, under "API", select "Create New API Token" and 
       a file "kaggle.json" will be downloaded on your computer.
    4. Move that "kaggle.json" file to the following path: 
       "C:\Users\<username>\.kaggle".
    5. Run this method."""
    # Authenticate to Kaggle:
    try:
        api = KaggleApi()
        api.authenticate()
    except IOError:
        print(setup_help_msg)
        raise

    # Download and unzip the CSV files from Kaggle:
    if ask_download_confirmation:
        download_consent = input("Warning: 2.47 GB of CSV files will be "
                                 "downloaded from Kaggle. "
                                 "Is this OK? Type 'y' to continue:")
        if download_consent != 'y':
            raise RuntimeError("Stopped creating the houses dataset, "
                               "you did not consent to download it.")
    dataset = 'msorondo/argentina-venta-de-propiedades'
    # api.dataset_list_files(dataset) is buggy + we're discarding one file
    # (ar_properties.csv), so let's specify them manually:
    csv_files = [
        'uy_properties_crude.csv',  # smallest first for debugging
        'ar_properties_crude.csv',
        'co_properties_crude.csv',
        'ec_properties_crude.csv',
        'pe_properties_crude.csv',
        #'uy_properties_crude.csv'
    ]
    os.makedirs(data_folder, exist_ok=True)
    for csv_file in tqdm(csv_files,
                         desc="Downloading CSV files of the Kaggle dataset..."):
        api.dataset_download_file(dataset, csv_file, data_folder)
        zip_file = os.path.join(data_folder, csv_file + ".zip")
        with ZipFile(zip_file) as zf:
            zf.extract(csv_file, data_folder)
        os.remove(zip_file)

    # Combine all CSVs into 1 big dataframe
    # & add the country of each loaded CSV file:
    print("Combining the CSVs into one basetable...")
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_folder, csv_file))
        country = csv_file.split("_")[0]
        df["country"] = country
        dfs.append(df)
    basetable = pd.concat(dfs, axis=0, ignore_index=True)
    del dfs

    # Keep only houses for sale. (Some are paid for per month, we assume those
    # are houses for rent and less applicable in that case, for this toy
    # dataset).
    print("Filtering only houses for sale...")
    basetable = basetable[basetable.price_period.isna()]

    # The houses are from 5 different countries, with different currencies,
    # so create one target column (house price) in a single currency (EUR):
    print("Converting house prices in different currencies to EUR...")

    def price_to_eur(price, currency):
        # 1 Argentine Peso equals 0,0092 Euro
        if currency == "ARS":
            return price * 0.0092
        # 1 United States Dollar equals 0,84 Euro
        elif currency == "USD":
            return price * 0.84
        elif pd.isna(currency):
            return np.nan
        # 1 Colombian Peso equals 0,00024 Euro
        elif currency == "COP":
            return price * 0.00024
        # 1 Sol equals 0,23 Euro
        elif currency == "PEN":
            return price * 0.23
        # 1 Uruguayan Peso equals 0,019 Euro
        elif currency == "UYU":
            return price * 0.019
        else:
            raise ValueError("Unexpected currency.")

    basetable["price_EUR"] = basetable[["price", "currency"]].apply(
        lambda row: price_to_eur(row[0], row[1]), axis=1)

    # Create a target column for a classification problem:
    # which houses are more expensive than 300K?
    # A Cobra model will then tell which features explain WHY this is the case.
    basetable["price_EUR_>300K"] = basetable.price_EUR > 300_000

    # Derived features for the datetime columns:
    print("Creating derived features for the datetime columns...")
    basetable["start_date"] = pd.to_datetime(basetable["start_date"],
                                             format="%Y-%m-%d",
                                             errors='coerce')  # avoid OutOfBoundsDatetime
    basetable["end_date"] = pd.to_datetime(basetable["end_date"],
                                           format="%Y-%m-%d",
                                           errors='coerce')
    basetable["created_on"] = pd.to_datetime(basetable["created_on"],
                                             format="%Y-%m-%d",
                                             errors='coerce')
    datetime_features = ["start_date", "end_date", "created_on"]
    derived_datetime_features = []
    for datetime_feature in datetime_features:
        basetable[datetime_feature + "_year"] = basetable[datetime_feature].dt.year
        basetable[datetime_feature + "_month"] = basetable[datetime_feature].dt.month
        basetable[datetime_feature + "_day"] = basetable[datetime_feature].dt.day
        basetable[datetime_feature + "_quarter"] = basetable[datetime_feature].dt.quarter
        derived_datetime_features += [
            datetime_feature + "_year",
            datetime_feature + "_month",
            datetime_feature + "_day",
            datetime_feature + "_quarter"
        ]

    # To reproduce the cobra performance issues under the same circumstances,
    # we need the dataframe to have 300 columns.
    print("Adding extra columns with random values, "
          "to reproduce bigger dataframes...")
    # => Add some irrelevant features (features with random values):
    num_random_features = 150
    random_feature_cols = [f"random_feature_{feat_idx}"
                           for feat_idx in range(num_random_features)]
    df_randoms = pd.DataFrame(
        np.random.rand(basetable.shape[0], num_random_features) * 100,
        columns=random_feature_cols)
    basetable = pd.concat([basetable, df_randoms], axis=1)
    print("Dataframe shape after adding irrelevant features:", basetable.shape)
    # => ... and add some correlated (in this case identical) features:
    random_feature_corr_cols = [col + "_corr" for col in random_feature_cols]
    df_corr = df_randoms.copy().rename(columns={
        random_feature_col: random_feature_corr_col
        for random_feature_col, random_feature_corr_col
        in zip(random_feature_cols, random_feature_corr_cols)
    })
    basetable = pd.concat([basetable, df_corr], axis=1)
    del df_randoms, df_corr
    print("Dataframe shape after adding correlated features:", basetable.shape)

    return basetable
