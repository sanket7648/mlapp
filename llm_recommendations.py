from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient model

def llm_based_recommendations(train_data, item_name, top_n=10):
    print(f"Using LLM to find recommendations for '{item_name}'.")

    # Generate embedding for the input product name
    input_embedding = model.encode(item_name, convert_to_tensor=True)

    # Generate embeddings for all product names in the dataset
    product_names = train_data['Name'].tolist()
    product_embeddings = model.encode(product_names, convert_to_tensor=True)

    # Calculate cosine similarity between input and all products
    similarities = util.cos_sim(input_embedding, product_embeddings)[0]

    # Add similarity scores to the dataset
    train_data['similarity'] = similarities.cpu().numpy()

    # Get the top N most similar products
    top_similar_items = train_data.sort_values(by='similarity', ascending=False).head(top_n)

    # Return the top N recommendations
    return top_similar_items[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]