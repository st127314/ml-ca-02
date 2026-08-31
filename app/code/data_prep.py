"""Data loading and cleaning, carried over from A1.

Kept in a module so the notebook and any script agree on exactly one definition of
"the cleaned dataset". Every correction here was justified in A1; the comments say
what each one is for so the reasoning travels with the code.
"""

import numpy as np
import pandas as pd

KM_DRIVEN_LIMIT = 500_000
RARE_BRAND_MIN_COUNT = 10
TWO_WORD_BRANDS = {"Land Rover", "Ashok Leyland"}
OWNER_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 5,
}


def extract_brand(name):
    """First word of the name, except for the two makes whose name is two words.
    Taking the first word blindly turns "Land Rover" into the brand "Land"."""
    words = str(name).split()
    leading_pair = " ".join(words[:2])
    return leading_pair if leading_pair in TWO_WORD_BRANDS else words[0]


def load_clean_data(path="../datasets/Cars.csv", verbose=True):
    df = pd.read_csv(path)
    raw_rows = len(df)

    df = df.drop_duplicates().copy()
    df = df.drop(columns=["torque"])
    df = df[df["owner"] != "Test Drive Car"].copy()
    df["owner"] = df["owner"].map(OWNER_MAP)
    # CNG/LPG record mileage in km/kg, a different unit from kmpl, so they are not
    # comparable with the rest of the column.
    df = df[~df["fuel"].isin(["CNG", "LPG"])].copy()

    for column in ["mileage", "engine", "max_power"]:
        df[column] = pd.to_numeric(
            df[column].astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False),
            errors="coerce",
        )
        # Zeros here mean "not recorded", not "measured as zero". SimpleImputer only
        # replaces NaN, so a 0.0 would train the model as a real measurement.
        df.loc[df[column] == 0, column] = np.nan

    # 2,360,457 km is about 59 times around the Earth: a typing error, not a car.
    df = df[df["km_driven"] <= KM_DRIVEN_LIMIT].copy()

    df["brand"] = df["name"].map(extract_brand)
    df = df.drop(columns=["name"])
    # A brand with one car becomes a one-hot column fitted to a single example.
    brand_counts = df["brand"].value_counts()
    df["brand"] = df["brand"].where(df["brand"].map(brand_counts) >= RARE_BRAND_MIN_COUNT, "Other")

    df = df.drop_duplicates().reset_index(drop=True)
    df = df[["brand"] + [c for c in df.columns if c != "brand"]]

    if verbose:
        print(f"raw rows: {raw_rows}  ->  cleaned rows: {len(df)}")
    return df
