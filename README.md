# LLM and RAG-based Dermatological Intervention Study System

This repository contains two key components for a system designed to analyze social media data and answer dermatological intervention questions. The system uses advanced large language models (LLMs) and RAG (Retrieval Augmented Generation) techniques. It extracts and visualizes relationships between drugs and side effects, based on large datasets of social media comments, and integrates with a conversational chatbot for responding to detailed queries about dermatological interventions.

## Features

### 1. **LLMGraphs.py**
- **Graph Creation for Drug-Side Effects**: Uses LangChain's LLM-based graph transformer to build a graph representing relationships between dermatological drugs (e.g., Rinvoq, Upadacitinib) and side effects (e.g., headache, nausea).
- **Custom Drug-Side Effect Graph**: Leverages filtered Reddit comments to construct a focused graph showing drug treatments and their associated side effects.
- **Dynamic Graph Generation**: Enables the creation of subgraphs for specific drugs, such as Rinvoq and Upadacitinib, and visualizes their relationships using NetworkX.
- **Statistics Extraction**: Extracts important statistics such as the number of diseases treated and side effects caused by each drug.
- **Social Media Data Processing**: Utilizes a dataset of social media comments to retrieve insights about specific drugs and their effects.
- **Graph Visualization**: Visualizes the graph using matplotlib, creating aesthetically enhanced visualizations for easier understanding of the relationships between drugs and their effects.

### 2. **RAGChatbot.py**
- **RAG-based Chatbot**: Implements a chatbot using Retrieval Augmented Generation to answer queries about dermatological interventions based on user inputs and a knowledge base.
- **Contextual Query Retrieval**: Combines social media data (from titles and descriptions) to build a rich knowledge base for answering queries about interventions.
- **FAISS Embeddings**: Uses FAISS for indexing social media post embeddings and quickly retrieving relevant posts based on similarity to user queries.
- **Gradio Web Interface**: A user-friendly interface built using Gradio that allows users to ask detailed questions and receive contextual responses.
- **Interactive Web-based Interface**: Provides a conversational experience for users with an intuitive input-output system.

## Technologies Used

### 1. **LangChain and LLMs**
- Uses LangChain's `LLMGraphTransformer` to extract entities and relationships from large datasets using OpenAI's GPT models. Graphs are created based on predefined nodes (e.g., drugs) and relationships (e.g., treats, causes).

### 2. **NetworkX for Graph Analysis**
- `NetworkxEntityGraph` is used to generate and manipulate graphs of relationships between drugs and side effects. The graphs are built dynamically from processed documents.

### 3. **FAISS for Similarity Search**
- FAISS is used in the RAGChatbot to efficiently search through high-dimensional embeddings of social media posts, enabling fast retrieval of relevant contexts.

### 4. **Sentence Transformers**
- Uses Sentence-BERT to embed social media posts into high-dimensional vectors for similarity searching.

### 5. **Gradio for Chatbot Interface**
- The Gradio interface provides a simple, web-based system for users to interact with the chatbot and get detailed answers to their dermatological intervention questions.

### 6. **OpenAI GPT-3.5 Turbo**
- GPT-3.5 is utilized in both the LLMGraphs and RAGChatbot components for generating responses and extracting graph-based relationships from the social media dataset.

## Visualizations

- **Drug-Specific Graphs**: Graphs showing relationships between drugs (e.g., Rinvoq, Upadacitinib) and their side effects are visualized using NetworkX and matplotlib.
  
![Rinvoq Graph](/mnt/data/file-n35jmgUtcTSLvwl0x4qSDPny)
![Upadacitinib Graph](/mnt/data/file-KaCRfdOqIZbbDfrf7E0RMFl3)

- **Chatbot Interface**: The Gradio-based chatbot interface allows users to ask questions and get detailed responses based on indexed knowledge from social media.
  
![SMEDIS Web Interface](https://i.postimg.cc/0yQVHn3W/Web-Interface.png)

## Usage

1. **Run LLMGraphs.py**
   - Install dependencies: `!pip install langchain-experimental langchain-community langchain networkx openai matplotlib json-repair`
   - Preprocess the data and run the script to visualize relationships between drugs and side effects.
   - Modify allowed drugs and side effects in the code to customize graph generation.

2. **Run RAGChatbot.py**
   - Install dependencies: `!pip install sentence-transformers faiss-cpu openai gradio`
   - Use the Gradio interface to ask questions about dermatological interventions and receive detailed, contextual responses.
   - Upload the required dataset for chatbot usage.

## Key Components

- **LLMGraphTransformer**: Extracts graph entities and relationships from social media data.
- **FAISS**: Performs similarity search on high-dimensional embeddings.
- **Gradio Interface**: Provides a web-based interface for user interaction.
- **OpenAI GPT Models**: Used for generating text-based responses and extracting relationships.

## Future Improvements

- Expand the list of allowed nodes (drugs, side effects) to include more detailed interventions.
- Add support for multilingual queries in the chatbot.
- Improve graph aesthetics by experimenting with different layouts and visualization techniques.
