!pip install langchain-experimental langchain-community langchain networkx langchain-core openai==0.28 matplotlib json-repair

import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chat_models import ChatOpenAI
import time
import networkx as nx
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from google.colab import drive
import pandas as pd

drive.mount('/content/drive')

comments_sample = pd.read_csv('/content/drive/MyDrive/SMEDIS/Full_Reddit_Data.csv')

OPENAI_API_KEY = ''
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Preprocessing: Fill NaN values and ensure all entries are strings
comments_sample['Comment Body'] = comments_sample['Comment Body'].fillna('')
comments_sample['Comment Body'] = comments_sample['Comment Body'].astype(str)

# Filter comments containing specific drug names
filtered_comments = comments_sample[comments_sample['Comment Body'].str.contains('Upadacitinib|Rinvoq', na=False, case=False)]
filtered_comments = filtered_comments.head(10000)
comments = ' '.join(filtered_comments['Comment Body'].astype(str))

# Function to check the number of tokens
def count_tokens(text):
    return len(text) / 4  # Approximate 1 token is equal to 4 characters

if count_tokens(comments) > 950000:
    print("Number of tokens exceeds 0.95 million. Stopping processing.")
else:
    test_chunk_size = 1000  # Chunk size for initial testing
    test_chunks = [' '.join(comments[i:i + test_chunk_size]) for i in range(0, len(comments), test_chunk_size)]

    # Limit to processing only the first 20 chunks to reduce API usage
    limited_documents = [Document(page_content=chunk) for chunk in test_chunks[:20]]

# Define allowed nodes and relationships
drugs = [
    'Upadacitinib', 'Rinvoq', 'Jak inhibitors', 'Abrocitinib', 'Cibinqo',
    'Baricitinib', 'Olumiant', 'Ritlecitinib', 'Litfulo', 'Tofacitinib',
    'Xeljanz', 'Filgotinib', 'Jyseleca', 'Deucravacitinib', 'Sotyktu',
    'Delgocitinib', 'Corectim', 'Ruxolitinib', 'Jakavi', 'Opzelura',
    'Peficitinib', 'Smyraf'
]

side_effects = [
    'nausea', 'headache', 'fatigue', 'diarrhea', 'infections', 'Bronchitis',
    'Common cold', 'Ear infection', 'Urinary tract infection', 'pneumonia',
    'high blood pressure', 'liver damage', 'blood clots', 'stomach pain',
    'tuberculosis'
]

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,  # Use the ChatOpenAI instance
    allowed_nodes=drugs + side_effects,
    allowed_relationships=['causes', 'treats']
)

cumulative_graph = NetworkxEntityGraph()

# Process documents and build the graph
for document in limited_documents:
    try:
        graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents([document])
        for node in graph_documents_filtered[0].nodes:
            cumulative_graph.add_node(node.id)
        for edge in graph_documents_filtered[0].relationships:
            cumulative_graph._graph.add_edge(edge.source.id, edge.target.id, relation=edge.type)
    except Exception as e:
        print("Failed to process document:", e)
        continue

print("Number of nodes:", cumulative_graph._graph.number_of_nodes())
print("Number of edges:", cumulative_graph._graph.number_of_edges())

import matplotlib.pyplot as plt

# Function to create subgraph for a specific drug
def create_drug_subgraph(cumulative_graph, drug_name):
    subgraph = nx.DiGraph()
    for node, data in cumulative_graph._graph.nodes(data=True):
        if data.get('name') == drug_name or drug_name in node:
            subgraph.add_node(node, **data)
            for edge in cumulative_graph._graph.edges(node, data=True):
                subgraph.add_edge(edge[0], edge[1], **edge[2])
                target_data = cumulative_graph._graph.nodes[edge[1]]
                subgraph.add_node(edge[1], **target_data)
    return subgraph

# Function to visualize graph with better aesthetics
def visualize_graph(nx_graph, title='Graph Visualization'):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(nx_graph, seed=42, k=0.15)  # Improved layout
    nx.draw_networkx_nodes(nx_graph, pos, node_color='skyblue', node_size=500, alpha=0.9)
    nx.draw_networkx_edges(nx_graph, pos, arrowstyle='-|>', arrowsize=15, edge_color='grey')
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_family='sans-serif')
    edge_labels = nx.get_edge_attributes(nx_graph, 'relation')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title(title, fontsize=15)
    plt.axis('off')
    plt.show()

# Create and visualize graphs for specific drugs
rinvoq_graph = create_drug_subgraph(cumulative_graph, 'Rinvoq')
upadacitinib_graph = create_drug_subgraph(cumulative_graph, 'Upadacitinib')

visualize_graph(rinvoq_graph, title='Rinvoq Graph')
visualize_graph(upadacitinib_graph, title='Upadacitinib Graph')

# Extract statistics from the graph
def extract_graph_statistics(graph):
    treats_count = 0
    causes_count = 0
    treats_diseases = set()
    causes_effects = set()

    for u, v, data in graph.edges(data=True):
        relation = data.get('relation', '')
        if relation == 'treats':
            treats_count += 1
            treats_diseases.add(v)
        elif relation == 'causes':
            causes_count += 1
            causes_effects.add(v)

    return {
        'treats_count': treats_count,
        'causes_count': causes_count,
        'treats_diseases': treats_diseases,
        'causes_effects': causes_effects
    }

# Extract and display statistics for each drug
rinvoq_stats = extract_graph_statistics(rinvoq_graph)
upadacitinib_stats = extract_graph_statistics(upadacitinib_graph)

print("\nStatistics for Rinvoq:")
print(f"Number of 'treats' relationships: {rinvoq_stats['treats_count']}")
print(f"Number of 'causes' relationships: {rinvoq_stats['causes_count']}")
print(f"Diseases treated by Rinvoq: {rinvoq_stats['treats_diseases']}")
print(f"Effects caused by Rinvoq: {rinvoq_stats['causes_effects']}")

print("\nStatistics for Upadacitinib:")
print(f"Number of 'treats' relationships: {upadacitinib_stats['treats_count']}")
print(f"Number of 'causes' relationships: {upadacitinib_stats['causes_count']}")
print(f"Diseases treated by Upadacitinib: {upadacitinib_stats['treats_diseases']}")
print(f"Effects caused by Upadacitinib: {upadacitinib_stats['causes_effects']}")

# Count occurrences in the original data
def count_occurrences(data, terms):
    term_counts = {term: 0 for term in terms}
    for comment in data:
        for term in terms:
            term_counts[term] += comment.lower().count(term.lower())
    return term_counts

# Count diseases and side effects occurrences in the original data
diseases_and_effects = list(rinvoq_stats['treats_diseases']) + list(rinvoq_stats['causes_effects']) + \
                       list(upadacitinib_stats['treats_diseases']) + list(upadacitinib_stats['causes_effects'])

occurrences = count_occurrences(comments_sample['Comment Body'], diseases_and_effects)

print("\nOccurrences of diseases and effects in the original data:")
for term, count in occurrences.items():
    print(f"{term}: {count} occurrences")

# Use GraphQA Chain for querying
chain = GraphQAChain.from_llm(llm=llm, graph=cumulative_graph, verbose=True)
question = "What does Rinvoq treat?"
response = chain.run(question)

print(response)
