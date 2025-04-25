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

# Fungsi untuk membuat visualisasi jaringan
def plot_network(G, centrality_metric, metric_name, color_scale, top_n=150):
    # Pilih top N nodes berdasarkan centrality metric
    top_nodes = sorted(centrality_metric, key=centrality_metric.get, reverse=True)[:top_n]
    
    # Buat subgraph hanya dengan top nodes
    G_sub = G.subgraph(top_nodes).copy()
    
    # Compute layout
    pos = nx.spring_layout(G_sub, k=1.5, iterations=300, seed=42)
    
    # Prepare data for Plotly
    edge_x = []
    edge_y = []
    for edge in G_sub.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G_sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>{metric_name}: {centrality_metric[node]:.4f}")
        node_size.append(centrality_metric[node] * 100)
        node_color.append(centrality_metric[node])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node if centrality_metric[node] > np.percentile([centrality_metric[n] for n in G_sub.nodes()], 90) else "" for node in G_sub.nodes()],
        textposition="top center",
        textfont=dict(size=12),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=color_scale,
            color=node_color,
            size=[max(10, s) for s in node_size],
            sizemode='diameter',
            sizeref=2*max(node_size)/(40**2),
            line=dict(width=1, color='#888'),
            colorbar=dict(
                thickness=15,
                title=f'{metric_name} Centrality',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<b>{metric_name} Centrality</b> – Top {top_n} Actors',
                        titlefont=dict(size=18),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(255,255,255,0.95)',
                        paper_bgcolor='rgba(255,255,255,0.95)',
                        height=600
                    ))
    
    return fig

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
    
    # Menghitung centrality metrics tanpa caching
    centrality_metrics = calculate_centrality(G)
    
    with tab1:
        st.header("Overview Jaringan Sosial")
        
        # Tampilkan statistik dengan visualisasi yang lebih menarik
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2 style="color: #4e73df; margin: 0;">{G.number_of_nodes():,}</h2>
                <p style="margin: 0;">Jumlah Aktor</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2 style="color: #1cc88a; margin: 0;">{G.number_of_edges():,}</h2>
                <p style="margin: 0;">Jumlah Interaksi</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col3:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2 style="color: #f6c23e; margin: 0;">{avg_degree:.2f}</h2>
                <p style="margin: 0;">Rata-rata Degree</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col4:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2 style="color: #e74a3b; margin: 0;">{nx.density(G):.5f}</h2>
                <p style="margin: 0;">Densitas Jaringan</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tambahkan visualisasi distribusi degree
        st.subheader("Distribusi Degree")
        degree_dist = sorted([d for n, d in G.degree()], reverse=True)
        fig_degree_dist = px.histogram(degree_dist, nbins=50,
                                    labels={'value': 'Degree', 'count': 'Jumlah Aktor'},
                                    title="Distribusi Degree dalam Jaringan")
        fig_degree_dist.update_layout(bargap=0.1)
        st.plotly_chart(fig_degree_dist, use_container_width=True)
    
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