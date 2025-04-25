import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io
from PIL import Image
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Jaringan Sosial Program Stunting",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS untuk styling aplikasi
def apply_custom_styling():
    st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 15px;
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e73df !important;
        color: white !important;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-container {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        margin: 0;
        color: #666;
    }
    .info-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Panggil fungsi styling
apply_custom_styling()

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
        with st.spinner('Loading data...'):
            # Coba membaca dari path relatif terlebih dahulu
            df = pd.read_csv("data/db_merge_with_sentiment_24apr_drop.csv")
            st.success("Data loaded successfully!")
        return df
    except FileNotFoundError:
        # Jika tidak ditemukan, tampilkan pesan error
        st.error("File data tidak ditemukan. Mohon cek lokasi file data.")
        return pd.DataFrame()  # Return DataFrame kosong

# Fungsi untuk membangun jaringan dari dataframe
@st.cache_data
def build_network(df):
    # Extract username dan in_reply_to_screen_name untuk membuat jaringan
    db_sna = df[["username", "in_reply_to_screen_name"]]
    
    # Jika data terlalu besar, lakukan sampling
    if len(db_sna) > 5000:
        db_sna = db_sna.sample(5000, random_state=42)
    
    # Membuat directed graph
    G = nx.DiGraph()
    
    # Menambahkan edge ke graf berdasarkan data dengan batasan
    edge_count = 0
    for index, row in db_sna.iterrows():
        if pd.notna(row["username"]) and pd.notna(row["in_reply_to_screen_name"]):
            G.add_edge(row["username"], row["in_reply_to_screen_name"])
            edge_count += 1
            # Batasi jumlah edge untuk performa
            if edge_count >= 1000:  # Batas 1000 edge
                break
    
    # Menghapus self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Jika masih terlalu besar, hanya ambil giant component
    if G.number_of_nodes() > 500:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G

# Fungsi untuk menghitung centrality metrics tanpa caching
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

# Fungsi untuk membuat visualisasi jaringan interaktif dengan Plotly
def plot_simple_network(G, centrality_metric, metric_name, colormap="Blues", top_n=50):
    """
    Membuat visualisasi jaringan yang ringan menggunakan matplotlib
    """
    # Pilih hanya top N nodes untuk mengurangi kompleksitas
    top_nodes = sorted(centrality_metric, key=centrality_metric.get, reverse=True)[:top_n]
    
    # Buat subgraph dengan top nodes
    G_sub = G.subgraph(top_nodes).copy()
    
    # Buat figure matplotlib
    plt.figure(figsize=(10, 7))
    
    # Gunakan layout yang lebih sederhana dengan iterasi terbatas
    pos = nx.spring_layout(G_sub, k=0.3, iterations=20, seed=42)
    
    # Persiapkan ukuran node (diskalakan berdasarkan centrality)
    node_sizes = [centrality_metric[node] * 5000 + 50 for node in G_sub.nodes()]
    
    # Siapkan warna node berdasarkan nilai centrality
    node_colors = [centrality_metric[node] for node in G_sub.nodes()]
    
    # Gambar edges dengan warna transparan untuk mengurangi visual clutter
    nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=0.5, edge_color='grey')
    
    # Gambar nodes
    nodes = nx.draw_networkx_nodes(
        G_sub, 
        pos, 
        node_size=node_sizes,
        node_color=node_colors, 
        cmap=plt.cm.get_cmap(colormap),
        alpha=0.8,
        linewidths=0.5,
        edgecolors='white'
    )
    
    # Tambahkan colorbar
    plt.colorbar(nodes, label=f'{metric_name} Centrality', shrink=0.7)
    
    # Tampilkan label hanya untuk top 10% node
    top_n_labels = int(len(G_sub.nodes()) * 0.1) + 1  # Minimal 1 label
    top_label_nodes = {node: node for i, node in 
                        enumerate(sorted(centrality_metric, key=centrality_metric.get, reverse=True)[:top_n_labels]) 
                        if node in G_sub.nodes()}
    
    # Tampilkan labels dengan posisi yang sedikit digeser untuk keterbacaan lebih baik
    label_pos = {node: (pos[node][0], pos[node][1] + 0.02) for node in top_label_nodes}
    nx.draw_networkx_labels(G_sub, label_pos, labels=top_label_nodes, font_size=9, font_weight='bold')
    
    # Hapus axis untuk tampilan bersih
    plt.axis('off')
    plt.title(f"Top {top_n} Aktor berdasarkan {metric_name} Centrality", size=13)
    plt.tight_layout()
    
    # Simpan plot ke BytesIO buffer untuk ditampilkan di Streamlit
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plt.close()  # Penting untuk menutup plot dan menghemat memori
    
    return buffer


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
    # Sidebar untuk filter
    st.sidebar.header("Network Settings")
    
    # Filter untuk top_n
    top_n = st.sidebar.slider(
        "Jumlah Aktor yang Ditampilkan", 
        min_value=50, 
        max_value=300, 
        value=150,
        step=10
    )
    
    # Membangun jaringan
    G = build_network(db_merge)
    
    # Menghitung centrality metrics tanpa caching
    centrality_metrics = calculate_centrality(G)
    
    # Tabs untuk berbagai analisis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview Jaringan", 
        "Degree Centrality", 
        "Betweenness Centrality", 
        "Closeness Centrality", 
        "Eigenvector Centrality"
    ])
    
    with tab1:
        st.header("Overview Jaringan Sosial")
        
        # Tampilkan statistik dengan visualisasi yang lebih menarik
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value" style="color: #4e73df;">{G.number_of_nodes():,}</p>
                <p class="metric-label">Jumlah Aktor</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value" style="color: #1cc88a;">{G.number_of_edges():,}</p>
                <p class="metric-label">Jumlah Interaksi</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value" style="color: #f6c23e;">{avg_degree:.2f}</p>
                <p class="metric-label">Rata-rata Degree</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <p class="metric-value" style="color: #e74a3b;">{nx.density(G):.5f}</p>
                <p class="metric-label">Densitas Jaringan</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tambahkan visualisasi distribusi degree
        st.subheader("Distribusi Degree")
        degree_dist = [d for n, d in G.degree()]
        fig_degree_dist = px.histogram(
            degree_dist, 
            nbins=50,
            labels={'value': 'Degree', 'count': 'Jumlah Aktor'},
            title="Distribusi Degree dalam Jaringan",
            color_discrete_sequence=['#4e73df']
        )
        fig_degree_dist.update_layout(
            bargap=0.1,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_degree_dist, use_container_width=True)
        
        # Tampilkan infografis jaringan
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
            <h3>Struktur Jaringan</h3>
            <p>Visualisasi jaringan menunjukkan adanya beberapa aktor sentral yang menjadi pusat diskusi mengenai program stunting.
            Jaringan ini bersifat <b>sparse</b> (tidak padat) yang menandakan bahwa diskusi tidak terjadi secara merata di antara semua aktor.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
            <h3>Tentang Analisis Jaringan Sosial</h3>
            <p><b>Social Network Analysis (SNA)</b> membantu mengidentifikasi:</p>
            <ul>
            <li><b>Aktor kunci</b>: Individu/organisasi yang berperan penting dalam jaringan</li>
            <li><b>Aliran informasi</b>: Bagaimana informasi menyebar di antara aktor</li>
            <li><b>Kelompok/komunitas</b>: Kumpulan aktor yang saling terhubung erat</li>
            <li><b>Peran struktural</b>: Posisi dan fungsi aktor dalam jaringan</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
            <h3>Interpretasi Metrik Centrality</h3>
            <ul>
            <li><b>Degree Centrality</b>: Mengukur jumlah koneksi langsung yang dimiliki aktor</li>
            <li><b>Betweenness Centrality</b>: Mengukur peran aktor sebagai perantara/jembatan</li>
            <li><b>Closeness Centrality</b>: Mengukur jarak aktor ke semua aktor lain</li>
            <li><b>Eigenvector Centrality</b>: Mengukur koneksi aktor dengan aktor berpengaruh lain</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
            <h3>Manfaat untuk Program Stunting</h3>
            <p>Analisis jaringan sosial untuk program stunting dapat membantu:</p>
            <ul>
            <li>Mengidentifikasi aktor kunci untuk kampanye edukasi</li>
            <li>Menemukan jalur penyebaran informasi yang efektif</li>
            <li>Menentukan strategi komunikasi yang optimal</li>
            <li>Memahami dinamika diskusi publik tentang program stunting</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Degree Centrality")
        st.markdown("""
        <div class="info-card">
        <p><b>Degree Centrality</b> mengukur jumlah koneksi langsung yang dimiliki oleh setiap aktor dalam jaringan.
        Aktor dengan degree centrality tinggi memiliki banyak interaksi langsung dan sering menjadi pusat diskusi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi jaringan untuk degree centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Kontrol untuk mengatur jumlah node
            default_nodes = min(50, G.number_of_nodes())
            display_nodes = st.slider("Jumlah node yang ditampilkan:", 10, 100, default_nodes, key="degree_nodes")
            
            # Tampilkan visualisasi ringan
            buffer = plot_simple_network(G, centrality_metrics["degree"], "Degree", "Blues", display_nodes)
            st.image(buffer, use_column_width=True)
            
            # Tambahkan tombol download visualisasi
            btn = st.download_button(
                label="Download Visualisasi",
                data=buffer,
                file_name=f"degree_centrality_{display_nodes}_nodes.png",
                mime="image/png"
            )
        
        with col2:
            st.subheader("Top Aktor berdasarkan Degree Centrality")
            df_degree = display_top_nodes(centrality_metrics["degree"], "Degree")
            st.dataframe(df_degree.style.background_gradient(cmap="Blues"), hide_index=True, use_container_width=True)
            
            st.markdown("""
            <div class="info-card">
            <h4>Interpretasi</h4>
            <p>Aktor dengan degree centrality tinggi:</p>
            <ul>
            <li>Memiliki banyak interaksi langsung</li>
            <li>Sering menjadi pusat diskusi</li>
            <li>Memiliki pengaruh langsung yang besar</li>
            <li>Berperan sebagai penyebar informasi utama</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    
    with tab3:
        st.header("Betweenness Centrality")
        st.markdown("""
        <div class="info-card">
        <p><b>Betweenness Centrality</b> mengukur seberapa sering suatu aktor berada di jalur terpendek antara dua aktor lainnya.
        Aktor dengan betweenness centrality tinggi berperan sebagai 'jembatan' atau perantara dalam jaringan.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi jaringan untuk betweenness centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_nodes = st.slider("Jumlah node yang ditampilkan:", 10, 100, default_nodes, key="betweenness_nodes")
            buffer = plot_simple_network(G, centrality_metrics["betweenness"], "Betweenness", "Oranges", display_nodes)
            st.image(buffer, use_column_width=True)
            
            # Tambahkan tombol download visualisasi
            btn = st.download_button(
                label="Download Visualisasi",
                data=buffer,
                file_name=f"betweenness_centrality_{display_nodes}_nodes.png",
                mime="image/png"
            )
        
        with col2:
            st.subheader("Top Aktor berdasarkan Betweenness Centrality")
            df_betweenness = display_top_nodes(centrality_metrics["betweenness"], "Betweenness")
            st.dataframe(df_betweenness.style.background_gradient(cmap="Oranges"), hide_index=True, use_container_width=True)
    
    # Implementasi untuk tab Closeness Centrality
    with tab4:
        st.header("Closeness Centrality")
        st.markdown("""
        <div class="info-card">
        <p><b>Closeness Centrality</b> mengukur seberapa dekat suatu aktor dengan semua aktor lainnya dalam jaringan.
        Aktor dengan closeness centrality tinggi dapat menyebarkan informasi dengan cepat karena memiliki jarak yang pendek
        ke banyak aktor lain.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi jaringan untuk closeness centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_nodes = st.slider("Jumlah node yang ditampilkan:", 10, 100, default_nodes, key="closeness_nodes")
            buffer = plot_simple_network(G, centrality_metrics["closeness"], "Closeness", "Greens", display_nodes)
            st.image(buffer, use_column_width=True)
            
            # Tambahkan tombol download visualisasi
            btn = st.download_button(
                label="Download Visualisasi",
                data=buffer,
                file_name=f"closeness_centrality_{display_nodes}_nodes.png",
                mime="image/png"
            )
        
        with col2:
            st.subheader("Top Aktor berdasarkan Closeness Centrality")
            df_closeness = display_top_nodes(centrality_metrics["closeness"], "Closeness")
            st.dataframe(df_closeness.style.background_gradient(cmap="Greens"), hide_index=True, use_container_width=True)
            
            st.markdown("""
            <div class="info-card">
            <h4>Interpretasi</h4>
            <p>Aktor dengan closeness centrality tinggi:</p>
            <ul>
            <li>Dapat menjangkau banyak aktor dengan cepat</li>
            <li>Efisien dalam penyebaran informasi</li>
            <li>Memiliki akses ke sumber informasi beragam</li>
            <li>Strategis untuk kampanye program stunting</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Implementasi untuk tab Eigenvector Centrality
    with tab5:
        st.header("Eigenvector Centrality")
        st.markdown("""
        <div class="info-card">
        <p><b>Eigenvector Centrality</b> mengukur pengaruh suatu aktor berdasarkan pengaruh aktor lain yang terhubung dengannya.
        Aktor dengan eigenvector centrality tinggi memiliki koneksi dengan aktor-aktor berpengaruh lainnya.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi jaringan untuk eigenvector centrality
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_nodes = st.slider("Jumlah node yang ditampilkan:", 10, 100, default_nodes, key="eigenvector_nodes")
            buffer = plot_simple_network(G, centrality_metrics["eigenvector"], "Eigenvector", "Purples", display_nodes)
            st.image(buffer, use_column_width=True)
            
            # Tambahkan tombol download visualisasi
            btn = st.download_button(
                label="Download Visualisasi",
                data=buffer,
                file_name=f"eigenvector_centrality_{display_nodes}_nodes.png",
                mime="image/png"
            )
        
        with col2:
            st.subheader("Top Aktor berdasarkan Eigenvector Centrality")
            df_eigenvector = display_top_nodes(centrality_metrics["eigenvector"], "Eigenvector")
            st.dataframe(df_eigenvector.style.background_gradient(cmap="Purples"), hide_index=True, use_container_width=True)
            
            st.markdown("""
            <div class="info-card">
            <h4>Interpretasi</h4>
            <p>Aktor dengan eigenvector centrality tinggi:</p>
            <ul>
            <li>Terhubung dengan aktor-aktor penting</li>
            <li>Memiliki pengaruh tidak langsung yang besar</li>
            <li>Berperan sebagai opinion leader</li>
            <li>Strategis untuk kolaborasi dalam program stunting</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Kesimpulan dan rekomendasi
    st.header("Kesimpulan dan Rekomendasi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>Kesimpulan</h3>
        <ol>
        <li><b>Struktur Jaringan</b>: Percakapan tentang program stunting membentuk jaringan dengan beberapa aktor kunci yang berperan sebagai pusat informasi dan penyebar diskusi.</li>
        <li><b>Aktor Berpengaruh</b>: Terdapat perbedaan aktor berpengaruh berdasarkan metrik centrality yang berbeda, menunjukkan peran yang beragam dalam jaringan.</li>
        <li><b>Pola Komunikasi</b>: Informasi tentang program stunting menyebar melalui jalur-jalur tertentu dengan aktor perantara yang memiliki peran strategis.</li>
        <li><b>Kelompok Diskusi</b>: Terbentuk beberapa kelompok diskusi yang saling terhubung melalui aktor-aktor dengan betweenness centrality tinggi.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Rekomendasi</h3>
        <ol>
        <li><b>Kerjasama dengan Aktor Kunci</b>: Melibatkan aktor-aktor dengan centrality tinggi dalam kampanye program stunting untuk meningkatkan jangkauan dan dampak.</li>
        <li><b>Strategi Komunikasi Bertarget</b>: Mengembangkan strategi komunikasi yang berbeda untuk setiap jenis aktor berdasarkan perannya dalam jaringan.</li>
        <li><b>Penguatan Jaringan</b>: Meningkatkan konektivitas antar kelompok diskusi untuk memastikan informasi program stunting menyebar secara merata.</li>
        <li><b>Monitoring Percakapan</b>: Melakukan pemantauan berkelanjutan terhadap percakapan program stunting di media sosial untuk mengidentifikasi perubahan pola dan aktor berpengaruh.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Metodologi
    st.header("Metodologi")
    st.markdown("""
    <div class="info-card">
    <h3>Proses Analisis Jaringan Sosial</h3>
    <p><b>Analisis Jaringan Sosial (Social Network Analysis)</b> dilakukan dengan langkah-langkah berikut:</p>
    
    <ol>
    <li><b>Pengumpulan Data</b>: Data percakapan tentang program stunting dikumpulkan dari media sosial X.</li>
    
    <li><b>Pemodelan Jaringan</b>: Interaksi antar pengguna dimodelkan sebagai directed graph, dengan username sebagai nodes dan reply sebagai edges.</li>
    
    <li><b>Analisis Centrality</b>: Berbagai metrik centrality (degree, betweenness, closeness, eigenvector) dihitung untuk mengidentifikasi aktor-aktor kunci.</li>
    
    <li><b>Visualisasi</b>: Jaringan divisualisasikan secara interaktif dengan ukuran node proporsional terhadap nilai centrality untuk memudahkan interpretasi.</li>
    
    <li><b>Interpretasi</b>: Hasil analisis diinterpretasikan untuk mengidentifikasi peran dan posisi aktor dalam jaringan serta pola aliran informasi.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.warning("Data tidak tersedia. Pastikan file data telah tersedia di lokasi yang benar.")