"""
Dataset 2: Customer Personality Analysis -> clean + engineer features.
Target: Response (accepted the last marketing campaign, 1/0)
"""
import pandas as pd
import numpy as np

df = pd.read_csv("marketing_campaign.csv", sep="\t")
print("Raw shape:", df.shape)
print(df.isna().sum()[df.isna().sum() > 0])

# --- Cleaning ---
df = df.dropna(subset=["Income"])
df = df[df["Income"] < 200000]  # remove extreme outliers
df = df[df["Year_Birth"] > 1930]  # remove implausible birth years (e.g. 1893, 1899, 1900)

# --- Feature engineering ---
df["Age"] = 2014 - df["Year_Birth"]
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days

mnt_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df["Total_Spend"] = df[mnt_cols].sum(axis=1)

purchase_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
df["Total_PriorCampaignsAccepted"] = df[cmp_cols].sum(axis=1)

df["Children"] = df["Kidhome"] + df["Teenhome"]

# simplify marital status / education
df["Marital_Status"] = df["Marital_Status"].replace(
    {"Alone": "Single", "YOLO": "Single", "Absurd": "Single"}
)

# drop redundant / constant columns
drop_cols = ["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"]
df = df.drop(columns=drop_cols)

df.to_csv("cpa_features.csv", index=False)
print("\nSaved cpa_features.csv, shape:", df.shape)
print("\nResponse balance:")
print(df["Response"].value_counts(normalize=True))
