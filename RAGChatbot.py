!pip install sentence-transformers
!pip install faiss-cpu
!pip install transformers
!pip install openai==0.28
!pip install gradio

import openai
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import gradio as gr
import os
from google.colab import files

# Upload the file from your local machine
uploaded = files.upload()

# Read the CSV file
df = pd.read_csv(list(uploaded.keys())[0])

# View the first few rows of the dataset
df.head()

openai.api_key = ""

# Combine 'Post Title' and 'Post Description' into a single context field
df['Context'] = df['Post Title'].astype(str) + " " + df['Post Description'].astype(str)

# Keep only the necessary columns: 'Context' and 'Comments'
knowledge_base = df[['Context', 'Comments']].drop_duplicates().reset_index(drop=True)

# Check the structure of the processed dataset
knowledge_base.head()

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the 'Context' field
context_embeddings = model.encode(knowledge_base['Context'].tolist())

# Set up a FAISS index to store the embeddings
embedding_dim = context_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Add the context embeddings to the FAISS index
faiss_index.add(np.array(context_embeddings))

# Check the number of embeddings indexed
print(f"Number of embeddings in the FAISS index: {faiss_index.ntotal}")

# Function to retrieve top-k similar contexts based on user query
def retrieve_similar_contexts(query, index, model, knowledge_base, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return knowledge_base.iloc[indices[0]]

# Function to generate a response using OpenAI GPT-3.5-turbo
def generate_response_with_gpt3(query, index, model, knowledge_base, top_k=5):
    # Retrieve top-k most relevant contexts
    relevant_contexts = retrieve_similar_contexts(query, index, model, knowledge_base, top_k=top_k)

    # Combine the top-k contexts into a single string
    combined_contexts = "\n\n".join(relevant_contexts['Context'].tolist())

    # Create the prompt for the chatbot
    prompt = (
        f"You are an expert agent that answers detailed questions based on the given context. "
        f"Answer the following question thoroughly, providing a detailed explanation.\n\n"
        f"The User Question Is: {query}\n\n"
        f"The context in which previous users are providing their opinions is:\n{combined_contexts}\n\n"
        f"Use the Context to Provide a detailed and insightful answer to the question."
    )

    # Generate response using OpenAI's GPT-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.7
    )

    # Return the generated response
    return response['choices'][0]['message']['content']

# Upload the image from your local machine
uploaded = files.upload()

# Get the file path of the uploaded image
image_path = list(uploaded.keys())[0]

# Define a Gradio interface for the chatbot
def chatbot(query):
    # Use the function to generate a response based on the user's query
    return generate_response_with_gpt3(query, faiss_index, model, knowledge_base)

# Create a Gradio interface with a textbox for the query and text output for the response
with gr.Blocks() as iface:
    # Main interface layout
    gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="text",
        title="Social Media Dermatological Intervention Study (SMEDIS)",
        description="Ask me anything about dermatological interventions based on social media data. I will retrieve the most relevant information and provide an insightful response."
    )

    # Add the logo and position it at the bottom-right using CSS
    with gr.Row():
        gr.Image(value=image_path, label="SMEDIS Logo", type="filepath", elem_id="bottom-right-logo")

    # CSS for placing the logo in the bottom-right corner
    iface.css = """
    #bottom-right-logo {
        position: fixed;
        bottom: 10px;
        left: 10px;
        width: 100px;
    }
    """

# Launch the Gradio interface
iface.launch()
