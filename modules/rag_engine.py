import os
import google.generativeai as genai
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_simple_response(question, df):

    question = question.lower()

    # Total revenue
    if "total revenue" in question:
        revenue = df["revenue"].sum()
        return f"The total revenue is SGD {revenue:,.2f}."

    # Best category
    elif "best category" in question or "highest revenue category" in question:
        category = (
            df.groupby("category")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )

        revenue = (
            df.groupby("category")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .iloc[0]
        )

        return f"{category} generated the highest revenue with SGD {revenue:,.2f}."

    # Top product
    elif "top product" in question or "best product" in question:

        product = (
            df.groupby("product_name")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )

        quantity = (
            df.groupby("product_name")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .iloc[0]
        )

        return f"The top-selling product is {product} with {quantity} units sold."

    else:
        return "Sorry, I can only answer questions related to sales analysis at the moment."
    
def generate_knowledge_chunks(df):
    chunks = []

    total_revenue = df["revenue"].sum()
    total_quantity = df["quantity"].sum()

    product_sales = (
        df.groupby("product_name")
        .agg(total_quantity=("quantity", "sum"), total_revenue=("revenue", "sum"))
        .sort_values(by="total_revenue", ascending=False)
    )

    category_sales = (
        df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    # 1. Overview
    chunks.append(
        f"Sales Overview: The total revenue is SGD {total_revenue:,.2f}, "
        f"and the total quantity sold is {int(total_quantity)} units."
    )

    # 2. Top product
    chunks.append(
        f"Product Performance: The top-selling product by quantity is "
        f"{product_sales.sort_values(by='total_quantity', ascending=False).index[0]} "
        f"with {int(product_sales.sort_values(by='total_quantity', ascending=False).iloc[0]['total_quantity'])} units sold."
    )

    # 3. Highest revenue product
    chunks.append(
        f"Revenue Performance: The highest revenue product is {product_sales.index[0]} "
        f"with SGD {product_sales.iloc[0]['total_revenue']:,.2f} revenue."
    )

    # 4. Best category
    chunks.append(
        f"Category Analysis: The highest revenue category is {category_sales.index[0]} "
        f"with SGD {category_sales.iloc[0]:,.2f} revenue."
    )

    # 5. Lowest category
    chunks.append(
        f"Improvement Area: The lowest revenue category is {category_sales.index[-1]} "
        f"with SGD {category_sales.iloc[-1]:,.2f} revenue. This category may require marketing attention."
    )

    # 6. Recommendation
    chunks.append(
        f"Business Recommendation: The seller should focus on promoting strong-performing products such as "
        f"{product_sales.index[0]} and improving weaker categories such as {category_sales.index[-1]}."
    )

    return chunks

def retrieve_relevant_chunks(question, chunks, top_k=3):
    question_words = question.lower().split()

    scored_chunks = []

    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = 0

        for word in question_words:
            if word in chunk_lower:
                score += 1

        scored_chunks.append((score, chunk))

    scored_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)

    relevant_chunks = [
        chunk for score, chunk in scored_chunks
        if score > 0
    ]

    return relevant_chunks[:top_k]


def generate_rag_response(question, chunks):
    retrieved_chunks = retrieve_relevant_chunks(question, chunks)

    if not retrieved_chunks:
        return "No relevant information was found based on the current dataset.", []

    context = "\n".join(retrieved_chunks)

    response = (
        "Based on the retrieved sales information, the system found the following insight:\n\n"
        f"{context}"
    )

    return response, retrieved_chunks

def generate_ai_rag_response(question, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information was found based on the current dataset."

    context = "\n".join(retrieved_chunks)

    prompt = f"""
    You are an AI e-commerce sales analyst.

    Use only the retrieved context below to answer the user's question.
    Do not invent numbers that are not provided.

    Your task:
    1. Answer the user's question clearly.
    2. Explain what the result means for the business.
    3. Provide one practical recommendation if relevant.
    4. If the retrieved context is insufficient, say that the dataset does not provide enough evidence.

    User question:
    {question}

    Retrieved context:
    {context}

    Provide a professional business-style answer.
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        result = model.generate_content(prompt)
        return result.text

    except Exception as e:
        return f"Gemini API error: {e}"

def build_vector_store(chunks):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="sales_knowledge"
    )

    # Clear old data to avoid duplicate chunks
    existing_items = collection.get()

    if existing_items["ids"]:
        collection.delete(ids=existing_items["ids"])

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return collection, embedding_model


def retrieve_chunks_from_vector_store(question, collection, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([question]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_chunks = results["documents"][0]

    return retrieved_chunks


def generate_vector_rag_response(question, chunks):
    collection, embedding_model = build_vector_store(chunks)

    retrieved_chunks = retrieve_chunks_from_vector_store(
        question,
        collection,
        embedding_model
    )

    response = generate_ai_rag_response(
        question,
        retrieved_chunks
    )

    return response, retrieved_chunks