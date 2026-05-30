import pandas as pd

REQUIRED_COLUMNS = [
    "order_date",
    "product_name",
    "category",
    "quantity",
    "sales"
]
def validate_data(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

COLUMN_MAPPING = {
    "order_date": "order_date",
    "date": "order_date",

    "product_name": "product_name",
    "product": "product_name",

    "category": "category",

    "quantity": "quantity",
    "qty": "quantity",

    "sales": "sales",
    "revenue": "sales",
    "amount": "sales",

    "profit": "profit",

    "region": "region"
}

def clean_data(df):

    df.columns = [
    col.lower().strip().replace(" ", "_")
    for col in df.columns
    ]

    df = df.rename(
        columns={
            col: COLUMN_MAPPING[col]
            for col in df.columns
            if col in COLUMN_MAPPING
        }
    )
    
    validate_data(df)

    df = df.dropna()
    df = df.drop_duplicates()

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["quantity"] = pd.to_numeric(df["quantity"])
    df["sales"] = pd.to_numeric(df["sales"])

    df["revenue"] = df["sales"]

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