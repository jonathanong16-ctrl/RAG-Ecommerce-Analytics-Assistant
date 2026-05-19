import streamlit as st
import pandas as pd
from modules.data_processing import clean_data, get_kpis

from modules.dashboard import (
    create_monthly_sales_chart,
    create_category_revenue_chart,
    get_top_products_table
)

from modules.rag_engine import (
    generate_knowledge_chunks,
    generate_vector_rag_response
)

st.set_page_config(page_title="RAG E-commerce Assistant", layout="wide")

st.title("RAG-based E-commerce Sales Analysis Assistant")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Data", "Dashboard", "AI Assistant"]
)

if "df" not in st.session_state:
    st.session_state.df = None

if page == "Upload Data":
    st.header("Upload E-commerce Dataset")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.info("Required columns: order_date, product_name, category, quantity, unit_price")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = clean_data(df)
            st.session_state.df = df
            st.success("Dataset uploaded and processed successfully!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Load Sample Data"):
        df = pd.read_csv("data/sample_sales.csv")
        df = clean_data(df)
        st.session_state.df = df
        st.success("Sample dataset loaded successfully!")
        st.dataframe(df)

elif page == "Dashboard":
    st.header("Sales Dashboard")

    if st.session_state.df is None:
        st.warning("Please upload or load a dataset first.")
    else:
        df = st.session_state.df
        total_revenue, total_quantity, top_product, best_category = get_kpis(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"SGD {total_revenue:,.2f}")
        col2.metric("Total Quantity Sold", int(total_quantity))
        col3.metric("Top Product", top_product)
        col4.metric("Best Category", best_category)

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            monthly_chart = create_monthly_sales_chart(df)
            st.plotly_chart(monthly_chart, use_container_width=True)

        with col_right:
            category_chart = create_category_revenue_chart(df)
            st.plotly_chart(category_chart, use_container_width=True)

        st.subheader("Top Products")
        top_products = get_top_products_table(df)
        st.dataframe(top_products, use_container_width=True)

        st.subheader("Processed Dataset")
        st.dataframe(df, use_container_width=True)

elif page == "AI Assistant":

    st.header("AI Sales Assistant")

    if st.session_state.df is None:
        st.warning("Please upload or load a dataset first.")

    else:
        st.subheader("Generated Knowledge Chunks")

        chunks = generate_knowledge_chunks(
            st.session_state.df
        )

    with st.expander("Generated Knowledge Chunks"):
        for i, chunk in enumerate(chunks, start=1):
            st.write(f"{i}. {chunk}")

    st.subheader("Ask Questions About Your Sales Data")

    question = st.text_input(
        "Enter your question",
         placeholder="Example: Which category generated the highest revenue?"
        )

    st.markdown("### Suggested Questions")

    st.markdown("""
        - What is the total revenue?
        - Which category generated the highest revenue?
        - What is the top-selling product?
        """)
    st.caption("You can ask any question related to the uploaded sales dataset. The examples below are only suggested questions.")

    if st.button("Generate Response"):

            if question.strip() == "":
                st.warning("Please enter a question.")

            else:
                response, retrieved_chunks = generate_vector_rag_response(
                    question,
                    chunks
                 )

                st.success("AI Response Generated")

                st.markdown("### Response")
                st.write(response)

                st.markdown("### Retrieved Context")
                for i, chunk in enumerate(retrieved_chunks, start=1):
                  st.info(f"{i}. {chunk}")

