import pandas as pd

REQUIRED_COLUMNS = ["order_date", "product_name", "category", "quantity", "unit_price"]

def validate_data(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

def clean_data(df):
    validate_data(df)

    df = df.dropna()
    df = df.drop_duplicates()

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["quantity"] = pd.to_numeric(df["quantity"])
    df["unit_price"] = pd.to_numeric(df["unit_price"])

    df["revenue"] = df["quantity"] * df["unit_price"]

    return df

def get_kpis(df):
    total_revenue = df["revenue"].sum()
    total_quantity = df["quantity"].sum()

    top_product = (
        df.groupby("product_name")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    best_category = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    return total_revenue, total_quantity, top_product, best_category