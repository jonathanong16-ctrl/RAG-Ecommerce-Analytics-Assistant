import pandas as pd
import plotly.express as px


def create_monthly_sales_chart(df: pd.DataFrame):
    df = df.copy()
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    monthly_sales = (
        df.groupby("month")["revenue"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="month",
        y="revenue",
        markers=True,
        title="Monthly Sales Trend"
    )

    return fig


def create_category_revenue_chart(df: pd.DataFrame):
    category_sales = (
        df.groupby("category")["revenue"]
        .sum()
        .reset_index()
        .sort_values(by="revenue", ascending=False)
    )

    fig = px.bar(
        category_sales,
        x="category",
        y="revenue",
        title="Revenue by Category"
    )

    return fig


def get_top_products_table(df: pd.DataFrame, top_n: int = 10):
    top_products = (
        df.groupby("product_name")
        .agg(
            total_quantity=("quantity", "sum"),
            total_revenue=("revenue", "sum")
        )
        .reset_index()
        .sort_values(by="total_revenue", ascending=False)
        .head(top_n)
    )

    return top_products