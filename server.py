from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import numpy as np
import os
import time


name = 'model'
# model_name = os.path.join('.', name)
model_name="sentence-transformers/all-MiniLM-L12-v2"
embeddings = np.load('description_clean.npy')
df_max = pd.read_csv(os.path.join('.','data','flipkart_com-ecommerce_sample.csv'))

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embedding(sentence):
    if pd.isna(sentence):  # Check for NaN or None
        return np.zeros(model.config.hidden_size)  # Return a zero vector of the same size as the embeddings
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    return sentence_embedding[0].numpy()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 > 0 and norm2 > 0:
        cosine_sim = dot_product / (norm1 * norm2)
    else:
        cosine_sim = 0.0
    return cosine_sim
     

def search(query, similarity_threshold):
    start_time = time.time()
    # This is a placeholder for the actual search function
    # data = {
    #     'Brand': ['Brand A', 'Brand B', 'Brand A', 'Brand C', 'Brand B'],
    #     'Product': ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5'],
    #     'Rating': [4.5, 4.0, 4.2, 3.8, 4.7],
    #     'Price': [100, 200, 150, 300, 250],
    #     'Availability': ['In Stock', 'Out of Stock', 'In Stock', 'In Stock', 'Out of Stock']
    # }
    search_query = get_sentence_embedding(query)
    # In your actual implementation, the similarity threshold will affect how results are filtered
    # For demonstration, this dummy implementation does not use the similarity threshold
    similarities = np.array([cosine_similarity(search_query, emb) for emb in embeddings])
    indices_above_threshold = np.where(similarities > similarity_threshold)[0]
    filtered_df = df_max.iloc[indices_above_threshold][["brand","product_name","description","product_rating"]]
    filtered_df['similarity'] = similarities[indices_above_threshold]
    filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
    end_time = time.time()
    response_time = end_time - start_time
    
    return filtered_df, response_time


# --------------------------- FRONTEND------------------------------------

st.title('Pranav Sanand Puligandla 20214548 MNNIT')

def create_redirect_button(url, button_text):
    button_html = f"""
    <a href="{url}" target="_blank">
        <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            {button_text}
        </button>
    </a>
    """
    st.markdown(button_html, unsafe_allow_html=True)

create_redirect_button("https://sites.google.com/view/eda-zepto/home?authuser=0", "Click Here for EDA")
st.title('Product Search and Filter')
similarity_threshold = st.slider('Select Similarity Threshold (For Developers, this value should be adjusted in production without the user knowing)', min_value=0.0, max_value=1.0, value=0.3)
# Search box
search_query = st.text_input("Enter your search query:")



if search_query:
    df, response_time = search(search_query, similarity_threshold)
    st.write(f"Query response time: {response_time:.4f} seconds")
    
    df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
    
    brand_filter = st.empty()
    rating_filter = st.empty()
    include_unrated_checkbox = st.empty()


    def apply_filters(dataframe, selected_brands, min_rating, max_rating, include_unrated):
        filtered_df = dataframe.copy()
        if selected_brands:
            filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
        
        if include_unrated:
            rating_mask = (
                (filtered_df['product_rating'] >= min_rating) & 
                (filtered_df['product_rating'] <= max_rating) |
                filtered_df['product_rating'].isna()
            )
        else:
            rating_mask = (
                (filtered_df['product_rating'] >= min_rating) & 
                (filtered_df['product_rating'] <= max_rating)
            )
        filtered_df = filtered_df[rating_mask]
        
        return filtered_df
    min_rating, max_rating = 0.0, 5.0
    filtered_df = apply_filters(df, [], min_rating, max_rating,include_unrated=True)

    all_brands = filtered_df['brand'].dropna().unique()
    brands = brand_filter.multiselect('Filter by Brand', options=all_brands,placeholder="All Brands")
    
    actual_min_rating = filtered_df['product_rating'].min()
    actual_max_rating = filtered_df['product_rating'].max()
    
    slider_min = max(0.0, actual_min_rating if not np.isnan(actual_min_rating) else 0.0)
    slider_max = min(5.0, actual_max_rating if not np.isnan(actual_max_rating) else 5.0)
    
    min_rating, max_rating = rating_filter.slider('Filter by Rating', 
                                                  min_value=slider_min, 
                                                  max_value=slider_max, 
                                                  value=(slider_min, slider_max))

    include_unrated = include_unrated_checkbox.checkbox("Include unrated products", value=True)

    filtered_df = apply_filters(df, brands, min_rating, max_rating, include_unrated)
    st.dataframe(filtered_df,hide_index=True)