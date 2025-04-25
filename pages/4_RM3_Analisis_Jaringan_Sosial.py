import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
import plotly.graph_objects as go
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Jaringan Sosial Program Stunting",
    page_icon="🔍",
    layout="wide"
)

# Judul halaman
st.title("🔍 RM3: Analisis Jaringan Sosial Program Stunting")

# Penjelasan singkat
with st.container():
    st.markdown("""
    ## Posisi dan Peran Aktor dalam Jaringan Sosial
    
    Halaman ini menyajikan hasil analisis jaringan sosial (Social Network Analysis/SNA) yang terbentuk dari percakapan 
    mengenai program stunting di media sosial X. Analisis ini membantu mengidentifikasi aktor-aktor kunci dan 
    pola interaksi dalam diskusi seputar program stunting di Indonesia.
    """)

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        # Path dapat disesuaikan sesuai kebutuhan
        data_path = "data/data_gephi2_5April2025.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Data dummy jika file tidak tersedia
            df = pd.DataFrame({
                "Source": ["user1", "user2", "user3", "user1", "user4"] * 10,
                "Target": ["user2", "user3", "user4", "user5", "user1"] * 10
            })
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame({"Source": [], "Target": []})

# Fungsi untuk membuat graph
def create_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["Source"], row["Target"])
    return G

# Fungsi untuk menghitung centrality dan top nodes
def calculate_centrality(G, centrality_type, top_n=50):
    if centrality_type == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_type == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == "closeness":
        centrality = nx.closeness_centrality(G)
    elif centrality_type == "eigenvector":
        # Convert to undirected for eigenvector centrality if it's a directed graph
        if nx.is_directed(G):
            G_und = G.to_undirected()
            centrality = nx.eigenvector_centrality(G_und, max_iter=1000, tol=1e-06)
        else:
            centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    else:
        raise ValueError(f"Unsupported centrality type: {centrality_type}")
    
    # Get top N nodes by centrality
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return centrality, top_nodes

# Fungsi untuk visualisasi jaringan dengan NetworkX
def visualize_network(G, centrality, top_n=150, cmap_name="Blues", title="Network Visualization"):
    # Create a copy of the graph and remove self-loops
    G_clean = G.copy()
    G_clean.remove_edges_from(nx.selfloop_edges(G_clean))
    
    # Get top N nodes by centrality
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_n]
    
    # Create subgraph of top nodes
    G_sub = G_clean.subgraph(top_nodes).copy()
    
    # Compute layout
    pos = nx.spring_layout(G_sub, k=1.0, iterations=200, seed=42)
    
    # Setup figure and node attributes
    fig, ax = plt.subplots(figsize=(12, 10))
    sizes = [centrality[n]*3000 for n in G_sub.nodes()]
    colors = [centrality[n] for n in G_sub.nodes()]
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(G_sub, pos,
                          node_size=sizes,
                          node_color=colors,
                          cmap=plt.cm.get_cmap(cmap_name),
                          alpha=0.9,
                          ax=ax)
    nx.draw_networkx_edges(G_sub, pos,
                          alpha=0.4, width=1,
                          ax=ax)
    
    # Label only top 30 nodes
    labels = {n: n for n in top_nodes[:30]}
    nx.draw_networkx_labels(G_sub, pos,
                           labels=labels,
                           font_size=10,
                           font_color="black",
                           ax=ax)
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    return fig

# Fungsi untuk visualisasi top nodes dengan Plotly
def visualize_top_nodes(top_nodes, title="Top Nodes by Centrality"):
    df = pd.DataFrame(top_nodes, columns=['Node', 'Centrality'])
    fig = px.bar(
        df, 
        x='Centrality', 
        y='Node', 
        orientation='h',
        title=title,
        labels={'Centrality': 'Centrality Value', 'Node': 'Username'},
        height=500
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

# Load data
df = load_data()

# Create graph
G = create_graph(df)

# Basic network stats
st.header("Statistik Jaringan")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jumlah Nodes", f"{G.number_of_nodes():,}")
with col2:
    st.metric("Jumlah Edges", f"{G.number_of_edges():,}")
with col3:
    st.metric("Kepadatan Jaringan", f"{nx.density(G):.4f}")
with col4:
    st.metric("Komponen Terhubung", f"{nx.number_weakly_connected_components(G)}")

# Centrality Analysis
st.header("Analisis Centralitas")

# Centrality type selector
centrality_type = st.selectbox(
    "Pilih Jenis Centralitas",
    ["degree", "betweenness", "closeness", "eigenvector"],
    format_func=lambda x: {
        "degree": "Degree Centrality", 
        "betweenness": "Betweenness Centrality",
        "closeness": "Closeness Centrality", 
        "eigenvector": "Eigenvector Centrality"
    }[x]
)

# Calculate selected centrality
centrality, top_nodes = calculate_centrality(G, centrality_type)

# Explanation of centrality measure
centrality_explanations = {
    "degree": """
    **Degree Centrality** mengukur jumlah koneksi langsung yang dimiliki oleh suatu node. 
    Node dengan degree centrality tinggi menunjukkan aktor yang aktif berinteraksi dengan banyak aktor lain dalam diskusi program stunting.
    """,
    
    "betweenness": """
    **Betweenness Centrality** mengukur seberapa sering suatu node berada di jalur terpendek antar dua node lainnya.
    Node dengan betweenness centrality tinggi berperan sebagai penghubung atau jembatan informasi dalam jaringan diskusi stunting.
    """,
    
    "closeness": """
    **Closeness Centrality** mengukur seberapa dekat suatu node dengan semua node lainnya dalam jaringan.
    Node dengan closeness centrality tinggi dapat menyebarkan informasi dengan cepat ke seluruh jaringan diskusi stunting.
    """,
    
    "eigenvector": """
    **Eigenvector Centrality** mengukur pengaruh suatu node berdasarkan pentingnya node-node yang terhubung dengannya.
    Node dengan eigenvector centrality tinggi terhubung dengan aktor-aktor penting lain dalam diskusi program stunting.
    """
}

st.markdown(centrality_explanations[centrality_type])

# Display the top nodes
st.subheader(f"Top 50 Nodes berdasarkan {centrality_type.capitalize()} Centrality")

# Create two columns for the visualization
col1, col2 = st.columns([3, 2])

with col1:
    # Network visualization
    cmap_dict = {
        "degree": "Blues",
        "betweenness": "Oranges",
        "closeness": "Greens",
        "eigenvector": "Purples"
    }
    
    title_dict = {
        "degree": "Degree Centrality – Top 150",
        "betweenness": "Betweenness Centrality – Top 150",
        "closeness": "Closeness Centrality – Top 150",
        "eigenvector": "Eigenvector Centrality – Top 150"
    }
    
    fig = visualize_network(
        G, 
        centrality, 
        top_n=150, 
        cmap_name=cmap_dict[centrality_type],
        title=title_dict[centrality_type]
    )
    st.pyplot(fig)

with col2:
    # Bar chart of top nodes
    bar_title = f"Top 20 Aktor berdasarkan {centrality_type.capitalize()} Centrality"
    bar_fig = visualize_top_nodes(top_nodes[:20], title=bar_title)
    st.plotly_chart(bar_fig, use_container_width=True)

# Show data table
st.subheader("Data Centralitas")
df_centrality = pd.DataFrame(top_nodes, columns=['Node', f'{centrality_type.capitalize()} Centrality'])
st.dataframe(df_centrality, use_container_width=True)

# Analisis Peran Aktor
st.header("Analisis Peran Aktor dalam Jaringan")

st.markdown("""
### Interpretasi Hasil Analisis

Berdasarkan perhitungan centrality, kita dapat mengidentifikasi beberapa peran kunci dalam jaringan percakapan mengenai program stunting:

1. **Influencers (Degree Centrality Tinggi)**
   - Aktor dengan koneksi langsung terbanyak yang dapat memengaruhi banyak pihak dalam diskusi
   - Berpotensi sebagai penyebar informasi dan opini tentang program stunting

2. **Brokers (Betweenness Centrality Tinggi)**
   - Aktor yang menjembatani kelompok-kelompok terpisah dalam diskusi
   - Berperan penting dalam penyebaran informasi antar komunitas yang berbeda

3. **Information Hubs (Closeness Centrality Tinggi)**
   - Aktor yang dapat menjangkau seluruh jaringan dengan cepat
   - Ideal untuk menyebarkan kampanye atau informasi program stunting secara efisien

4. **Strategic Connectors (Eigenvector Centrality Tinggi)**
   - Aktor yang terhubung dengan aktor-aktor penting lainnya
   - Memiliki pengaruh strategis dalam membentuk narasi tentang program stunting
""")

st.markdown("---")

# Conclusions
st.header("Kesimpulan dan Rekomendasi")

st.markdown("""
### Temuan Utama

1. Jaringan percakapan tentang program stunting di media sosial X menunjukkan struktur yang terpusat pada beberapa aktor kunci
2. Terdapat aktor-aktor yang berperan sebagai influencer, broker informasi, dan penghubung strategis
3. Pola interaksi menunjukkan adanya kelompok-kelompok diskusi yang terpisah namun terhubung melalui beberapa aktor penghubung

### Rekomendasi untuk Program Stunting

1. **Kolaborasi dengan Influencers**
   - Melibatkan aktor dengan degree centrality tinggi dalam kampanye edukasi stunting

2. **Penyebaran Informasi melalui Brokers**
   - Memanfaatkan aktor dengan betweenness centrality tinggi untuk menjangkau komunitas yang berbeda

3. **Peningkatan Respons melalui Information Hubs**
   - Memprioritaskan komunikasi dengan aktor dengan closeness centrality tinggi untuk respons cepat

4. **Pembentukan Narasi melalui Strategic Connectors**
   - Bekerja sama dengan aktor yang memiliki eigenvector centrality tinggi untuk membentuk narasi positif tentang program stunting
""")

# Reference and Footer
st.markdown("---")
st.caption("Analisis Jaringan Sosial Program Stunting di Media Sosial X | Data diperbarui per April 2025")