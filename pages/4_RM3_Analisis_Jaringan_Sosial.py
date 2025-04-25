import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from PIL import Image
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Jaringan Sosial Program Stunting",
    page_icon="🔍",
    layout="wide"
)

# Judul halaman
st.title("🔍 Analisis Jaringan Sosial pada Media Sosial X tentang Program Stunting")

# Deskripsi halaman
st.markdown("""
## Rumusan Masalah 3
**Menganalisis posisi dan peran aktor dalam jaringan sosial yang terbentuk dari percakapan mengenai program stunting di media sosial X**

Halaman ini menampilkan hasil analisis jaringan sosial (Social Network Analysis/SNA) untuk mengidentifikasi aktor-aktor 
kunci dalam percakapan mengenai program stunting di media sosial X. Analisis ini membantu memahami pola interaksi, 
pengaruh, dan peran berbagai aktor dalam diskusi publik tentang program stunting.
""")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        # Coba membaca dari path relatif terlebih dahulu
        df = pd.read_csv("data/db_merge_with_sentiment_24apr_drop.csv")
    except FileNotFoundError:
        # Jika tidak ditemukan, tampilkan pesan error
        st.error("File data tidak ditemukan. Mohon cek lokasi file data.")
        df = pd.DataFrame()  # Return DataFrame kosong
    return df

# Fungsi untuk membangun jaringan dari dataframe
@st.cache_data
def build_network(df):
    # Extract username dan in_reply_to_screen_name untuk membuat jaringan
    db_sna = df[["username", "in_reply_to_screen_name"]]
    
    # Membuat directed graph
    G = nx.DiGraph()
    
    # Menambahkan edge ke graf berdasarkan data
    for index, row in db_sna.iterrows():
        G.add_edge(row["username"], row["in_reply_to_screen_name"])
    
    # Menghapus self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

# Fungsi untuk menghitung centrality metrics
@st.cache_data
def calculate_centrality(G):
    # Menghitung berbagai centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Untuk eigenvector centrality, convert ke undirected graph
    G_und = G.to_undirected()
    eigenvector_centrality = nx.eigenvector_centrality(G_und, max_iter=1000, tol=1e-06)
    
    return {
        "degree": degree_centrality,
        "betweenness": betweenness_centrality,
        "closeness": closeness_centrality,
        "eigenvector": eigenvector_centrality
    }

# Fungsi untuk membuat visualisasi jaringan
def plot_network(G, centrality_metric, metric_name, color_map, top_n=150):
    # Pilih top N nodes berdasarkan centrality metric
    top_nodes = sorted(centrality_metric, key=centrality_metric.get, reverse=True)[:top_n]
    
    # Buat subgraph hanya dengan top nodes
    G_sub = G.subgraph(top_nodes).copy()
    
    # Compute layout
    pos = nx.spring_layout(G_sub, k=1.0, iterations=200, seed=42)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Node sizes dan colors berdasarkan centrality
    sizes = [centrality_metric[n]*3000 for n in G_sub.nodes()]
    colors = [centrality_metric[n] for n in G_sub.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_sub, pos,
        node_size=sizes,
        node_color=colors,
        cmap=color_map,
        alpha=0.9
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G_sub, pos,
        alpha=0.4, width=1
    )
    
    # Draw labels untuk top 30 nodes
    label_count = 30
    labels = {n: n for n in top_nodes[:label_count]}
    nx.draw_networkx_labels(
        G_sub, pos,
        labels=labels,
        font_size=12,
        font_color="black"
    )
    
    plt.title(f'{metric_name} Centrality – Top {top_n}', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    
    return plt

# Fungsi untuk menampilkan top nodes berdasarkan centrality
def display_top_nodes(centrality_dict, metric_name, n=50):
    # Sort nodes berdasarkan centrality value
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Convert ke DataFrame
    df = pd.DataFrame(sorted_nodes[:n], columns=['Node', f'{metric_name} Centrality'])
    
    return df

# Memuat data
db_merge = load_data()

if not db_merge.empty:
    # Tabs untuk berbagai analisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview Jaringan", 
        "Degree Centrality", 
        "Betweenness Centrality", 
        "Closeness Centrality", 
        "Eigenvector Centrality"
    ])
    
    # Membangun jaringan
    G = build_network(db_merge)
    
    # Menghitung centrality metrics
    centrality_metrics = calculate_centrality(G)
    
    with tab1:
        st.header("Overview Jaringan Sosial")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistik Jaringan")
            st.markdown(f"""
            - **Jumlah Nodes (Aktor)**: {G.number_of_nodes()}
            - **Jumlah Edges (Interaksi)**: {G.number_of_edges()}
            - **Rata-rata Degree**: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}
            - **Densitas Jaringan**: {nx.density(G):.5f}
            """)
            
            st.subheader("Interpretasi")
            st.markdown("""
            Jaringan sosial yang terbentuk dari percakapan tentang program stunting menunjukkan beberapa karakteristik penting:
            
            1. **Aktor Kunci**: Terdapat beberapa aktor dengan pengaruh tinggi yang menjadi pusat diskusi
            2. **Pola Interaksi**: Diskusi cenderung terkonsentrasi di sekitar aktor-aktor dengan centrality tinggi
            3. **Aliran Informasi**: Informasi tentang program stunting menyebar melalui jalur-jalur tertentu dalam jaringan
            """)
        
        with col2:
            st.subheader("Tentang Analisis Jaringan Sosial")
            st.markdown("""
            **Social Network Analysis (SNA)** membantu kita memahami struktur relasi antar aktor dalam jaringan sosial.
            Beberapa konsep penting dalam SNA:
            
            - **Centrality**: Mengukur posisi strategis aktor dalam jaringan
            - **Degree**: Jumlah koneksi langsung yang dimiliki aktor
            - **Betweenness**: Seberapa sering aktor menjadi 'jembatan' antar aktor lain
            - **Closeness**: Seberapa dekat aktor dengan semua aktor lain di jaringan
            - **Eigenvector**: Mengukur pengaruh aktor berdasarkan koneksinya dengan aktor berpengaruh lain
            """)
    
    with tab2:
        st.header("Degree Centrality")
        st.markdown("""
        **Degree Centrality** mengukur jumlah koneksi langsung yang dimiliki oleh setiap aktor dalam jaringan.
        Aktor dengan degree centrality tinggi memiliki banyak interaksi langsung dan sering menjadi pusat diskusi.
        """)
        
        # Visualisasi jaringan untuk degree centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_degree = plot_network(G, centrality_metrics["degree"], "Degree", plt.cm.Blues)
            st.pyplot(fig_degree)
        
        with col2:
            st.subheader("Top Aktor berdasarkan Degree Centrality")
            df_degree = display_top_nodes(centrality_metrics["degree"], "Degree")
            st.dataframe(df_degree, hide_index=True)
            
            st.subheader("Interpretasi")
            st.markdown("""
            Aktor dengan degree centrality tinggi:
            - Memiliki banyak interaksi langsung
            - Sering menjadi pusat diskusi
            - Memiliki pengaruh langsung yang besar
            - Berperan sebagai penyebar informasi utama
            """)
    
    with tab3:
        st.header("Betweenness Centrality")
        st.markdown("""
        **Betweenness Centrality** mengukur seberapa sering suatu aktor berada di jalur terpendek antara dua aktor lainnya.
        Aktor dengan betweenness centrality tinggi berperan sebagai 'jembatan' atau perantara dalam jaringan dan memiliki
        kontrol terhadap aliran informasi.
        """)
        
        # Visualisasi jaringan untuk betweenness centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_betweenness = plot_network(G, centrality_metrics["betweenness"], "Betweenness", plt.cm.Oranges)
            st.pyplot(fig_betweenness)
        
        with col2:
            st.subheader("Top Aktor berdasarkan Betweenness Centrality")
            df_betweenness = display_top_nodes(centrality_metrics["betweenness"], "Betweenness")
            st.dataframe(df_betweenness, hide_index=True)
            
            st.subheader("Interpretasi")
            st.markdown("""
            Aktor dengan betweenness centrality tinggi:
            - Menjadi perantara informasi
            - Menghubungkan kelompok-kelompok berbeda
            - Memiliki kontrol terhadap aliran informasi
            - Penting untuk penyebaran program stunting antar komunitas
            """)
    
    with tab4:
        st.header("Closeness Centrality")
        st.markdown("""
        **Closeness Centrality** mengukur seberapa dekat suatu aktor dengan semua aktor lainnya dalam jaringan.
        Aktor dengan closeness centrality tinggi dapat menyebarkan informasi dengan cepat karena memiliki jarak yang pendek
        ke banyak aktor lain.
        """)
        
        # Visualisasi jaringan untuk closeness centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_closeness = plot_network(G, centrality_metrics["closeness"], "Closeness", plt.cm.Greens)
            st.pyplot(fig_closeness)
        
        with col2:
            st.subheader("Top Aktor berdasarkan Closeness Centrality")
            df_closeness = display_top_nodes(centrality_metrics["closeness"], "Closeness")
            st.dataframe(df_closeness, hide_index=True)
            
            st.subheader("Interpretasi")
            st.markdown("""
            Aktor dengan closeness centrality tinggi:
            - Dapat menjangkau banyak aktor dengan cepat
            - Efisien dalam penyebaran informasi
            - Memiliki akses ke sumber informasi beragam
            - Strategis untuk kampanye program stunting
            """)
    
    with tab5:
        st.header("Eigenvector Centrality")
        st.markdown("""
        **Eigenvector Centrality** mengukur pengaruh suatu aktor berdasarkan pengaruh aktor lain yang terhubung dengannya.
        Aktor dengan eigenvector centrality tinggi memiliki koneksi dengan aktor-aktor berpengaruh lainnya.
        """)
        
        # Visualisasi jaringan untuk eigenvector centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_eigenvector = plot_network(G, centrality_metrics["eigenvector"], "Eigenvector", plt.cm.Purples)
            st.pyplot(fig_eigenvector)
        
        with col2:
            st.subheader("Top Aktor berdasarkan Eigenvector Centrality")
            df_eigenvector = display_top_nodes(centrality_metrics["eigenvector"], "Eigenvector")
            st.dataframe(df_eigenvector, hide_index=True)
            
            st.subheader("Interpretasi")
            st.markdown("""
            Aktor dengan eigenvector centrality tinggi:
            - Terhubung dengan aktor-aktor penting
            - Memiliki pengaruh tidak langsung yang besar
            - Berperan sebagai opinion leader
            - Strategis untuk kolaborasi dalam program stunting
            """)
    
    # Kesimpulan dan rekomendasi
    st.header("Kesimpulan dan Rekomendasi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Kesimpulan")
        st.markdown("""
        1. **Struktur Jaringan**: Percakapan tentang program stunting membentuk jaringan dengan beberapa aktor kunci yang berperan sebagai pusat informasi dan penyebar diskusi.
        
        2. **Aktor Berpengaruh**: Terdapat perbedaan aktor berpengaruh berdasarkan metrik centrality yang berbeda, menunjukkan peran yang beragam dalam jaringan.
        
        3. **Pola Komunikasi**: Informasi tentang program stunting menyebar melalui jalur-jalur tertentu dengan aktor perantara yang memiliki peran strategis.
        
        4. **Kelompok Diskusi**: Terbentuk beberapa kelompok diskusi yang saling terhubung melalui aktor-aktor dengan betweenness centrality tinggi.
        """)
    
    with col2:
        st.subheader("Rekomendasi")
        st.markdown("""
        1. **Kerjasama dengan Aktor Kunci**: Melibatkan aktor-aktor dengan centrality tinggi dalam kampanye program stunting untuk meningkatkan jangkauan dan dampak.
        
        2. **Strategi Komunikasi Bertarget**: Mengembangkan strategi komunikasi yang berbeda untuk setiap jenis aktor berdasarkan perannya dalam jaringan.
        
        3. **Penguatan Jaringan**: Meningkatkan konektivitas antar kelompok diskusi untuk memastikan informasi program stunting menyebar secara merata.
        
        4. **Monitoring Percakapan**: Melakukan pemantauan berkelanjutan terhadap percakapan program stunting di media sosial untuk mengidentifikasi perubahan pola dan aktor berpengaruh.
        """)
    
    # Metodologi
    st.header("Metodologi")
    st.markdown("""
    **Analisis Jaringan Sosial (Social Network Analysis)** dilakukan dengan langkah-langkah berikut:
    
    1. **Pengumpulan Data**: Data percakapan tentang program stunting dikumpulkan dari media sosial X.
    
    2. **Pemodelan Jaringan**: Interaksi antar pengguna dimodelkan sebagai directed graph, dengan username sebagai nodes dan reply sebagai edges.
    
    3. **Analisis Centrality**: Berbagai metrik centrality (degree, betweenness, closeness, eigenvector) dihitung untuk mengidentifikasi aktor-aktor kunci.
    
    4. **Visualisasi**: Jaringan divisualisasikan dengan ukuran node proporsional terhadap nilai centrality untuk memudahkan interpretasi.
    
    5. **Interpretasi**: Hasil analisis diinterpretasikan untuk mengidentifikasi peran dan posisi aktor dalam jaringan serta pola aliran informasi.
    """)
    
else:
    st.warning("Data tidak tersedia. Pastikan file data telah tersedia di lokasi yang benar.")