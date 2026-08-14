"""
Dataset 1: UCI Online Retail -> build customer-level RFM + behavioural features
and an engineered proxy target 'Engaged_Customer' (used as a stand-in for
"likely to respond to a targeted campaign" since this dataset has no
explicit campaign-response label).
"""
import pandas as pd
import numpy as np

df = pd.read_excel("OnlineRetail.xlsx")
print("Raw shape:", df.shape)

# --- Cleaning ---
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]  # remove cancellations
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["CustomerID"] = df["CustomerID"].astype(int)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("Cleaned shape:", df.shape)
print("Unique customers:", df["CustomerID"].nunique())

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    AvgBasketValue=("TotalPrice", "mean"),
    UniqueProducts=("StockCode", "nunique"),
    TotalItems=("Quantity", "sum"),
).reset_index()

# Dominant country per customer (behavioural/geographic feature)
top_country = df.groupby("CustomerID")["Country"].agg(lambda x: x.mode().iloc[0])
rfm = rfm.merge(top_country.rename("Country"), on="CustomerID")
rfm["IsUK"] = (rfm["Country"] == "United Kingdom").astype(int)

# Engineered proxy target: an "engaged" high-value repeat customer, i.e. the
# realistic marketing segment a retailer would target with a campaign.
freq_median = rfm["Frequency"].median()
mon_median = rfm["Monetary"].median()
rfm["Engaged_Customer"] = (
    (rfm["Frequency"] >= freq_median) & (rfm["Monetary"] >= mon_median)
).astype(int)

print("\nEngaged_Customer balance:")
print(rfm["Engaged_Customer"].value_counts(normalize=True))

rfm.to_csv("retail_rfm_features.csv", index=False)
print("\nSaved retail_rfm_features.csv, shape:", rfm.shape)
print(rfm.describe())
