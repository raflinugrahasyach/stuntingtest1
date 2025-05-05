import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Jaringan Sosial Program Stunting",
    page_icon="🔍",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .data-container {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563EB;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E40AF;
    }
    .gephi-viz {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Judul halaman
st.markdown("<div class='main-header'>🔍 Analisis Jaringan Sosial pada Media Sosial X tentang Program Stunting</div>", unsafe_allow_html=True)

# Deskripsi halaman
st.markdown("""
<div class='info-box'>
<h2>Rumusan Masalah 3</h2>
<strong>Menganalisis posisi dan peran aktor dalam jaringan sosial yang terbentuk dari percakapan mengenai program stunting di media sosial X</strong>

<p>Halaman ini menampilkan hasil analisis jaringan sosial (Social Network Analysis/SNA) untuk mengidentifikasi aktor-aktor 
kunci dalam percakapan mengenai program stunting di media sosial X. Analisis ini membantu memahami pola interaksi, 
pengaruh, dan peran berbagai aktor dalam diskusi publik tentang program stunting.</p>
</div>
""", unsafe_allow_html=True)

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
    db_sna = df[["username", "in_reply_to_screen_name"]].dropna()
    
    # Membuat directed graph
    G = nx.DiGraph()
    
    # Menambahkan edge ke graf berdasarkan data
    for index, row in db_sna.iterrows():
        G.add_edge(row["username"], row["in_reply_to_screen_name"])
    
    # Menghapus self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

# Fungsi untuk menghitung centrality metrics dengan penanganan error yang lebih baik
def calculate_centrality(G):
    # Menghitung berbagai centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, k=100, normalized=True)
    closeness_centrality = nx.closeness_centrality(G)
    
    # Eigenvector centrality untuk directed graph dengan penanganan error yang lebih baik
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
    except:
        try:
            # Fallback ke metode standar dengan iterasi yang lebih banyak
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-03)
        except:
            # Jika masih gagal, gunakan pendekatan PageRank sebagai pengganti
            st.warning("Eigenvector centrality calculation failed, using PageRank as an approximation.")
            eigenvector_centrality = nx.pagerank(G, alpha=0.85)
    
    return {
        "degree": degree_centrality,
        "betweenness": betweenness_centrality,
        "closeness": closeness_centrality,
        "eigenvector": eigenvector_centrality
    }

# Fungsi untuk menampilkan top nodes berdasarkan centrality
def display_top_nodes(centrality_dict, metric_name, n=10):
    # Sort nodes berdasarkan centrality value
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Convert ke DataFrame
    df = pd.DataFrame(sorted_nodes[:n], columns=['Aktor', f'{metric_name}'])
    df[f'{metric_name}'] = df[f'{metric_name}'].apply(lambda x: f"{x:.4f}")
    
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
    # Gunakan status spinner untuk menunjukkan proses
    with st.spinner('Menghitung metrics jaringan...'):
        centrality_metrics = calculate_centrality(G)
    
    with tab1:
        st.markdown("<div class='sub-header'>Overview Jaringan Sosial</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.subheader("Statistik Jaringan")
            
            # Metrics dengan styling
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            avg_degree = sum(dict(G.degree()).values()) / max(1, num_nodes)
            density = nx.density(G)
            
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric(label="Jumlah Aktor (Nodes)", value=f"{num_nodes:,}")
                st.metric(label="Rata-rata Koneksi", value=f"{avg_degree:.2f}")
            
            with col1b:
                st.metric(label="Jumlah Interaksi (Edges)", value=f"{num_edges:,}")
                st.metric(label="Densitas Jaringan", value=f"{density:.5f}")
            
            # Visualisasi proporsi
            component_sizes = [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
            
            if len(component_sizes) > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Hanya tampilkan top 5 komponen
                top_components = component_sizes[:5]
                others = sum(component_sizes[5:]) if len(component_sizes) > 5 else 0
                
                labels = [f"Komponen {i+1}: {size} aktor" for i, size in enumerate(top_components)]
                if others > 0:
                    top_components.append(others)
                    labels.append(f"Komponen lainnya: {others} aktor")
                
                ax.bar(range(len(top_components)), top_components, color=plt.cm.tab10.colors)
                ax.set_xticks(range(len(top_components)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Jumlah Aktor')
                ax.set_title('Distribusi Ukuran Komponen Jaringan', pad=20)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.subheader("Interpretasi")
            st.markdown("""
            Jaringan sosial yang terbentuk dari percakapan tentang program stunting menunjukkan beberapa karakteristik penting:
            
            1. **Aktor Kunci**: Terdapat beberapa aktor dengan pengaruh tinggi yang menjadi pusat diskusi
            2. **Pola Interaksi**: Diskusi cenderung terkonsentrasi di sekitar aktor-aktor dengan centrality tinggi
            3. **Aliran Informasi**: Informasi tentang program stunting menyebar melalui jalur-jalur tertentu dalam jaringan
            """)
            
            # Top aktor dari berbagai metrics
            st.subheader("Aktor-aktor Berpengaruh")
            
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.markdown("**Top 10 Degree Centrality:**")
                top_degree = display_top_nodes(centrality_metrics["degree"], "Nilai", 10)
                st.dataframe(top_degree, hide_index=True)
            
            with col2b:
                st.markdown("**Top 10 Betweenness Centrality:**")
                top_between = display_top_nodes(centrality_metrics["betweenness"], "Nilai", 10)
                st.dataframe(top_between, hide_index=True)
            
            # Visualisasi distribusi centrality
            st.subheader("Distribusi Nilai Centrality")
            
            centrality_values = {
                "Degree": list(centrality_metrics["degree"].values()),
                "Betweenness": list(centrality_metrics["betweenness"].values()),
                "Closeness": list(centrality_metrics["closeness"].values()),
                "Eigenvector": list(centrality_metrics["eigenvector"].values())
            }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            positions = range(4)
            violin_parts = ax.violinplot([centrality_values[k] for k in centrality_values.keys()], 
                                        positions=positions, 
                                        showmeans=True,
                                        widths=0.8)
            
            # Color customization
            for pc in violin_parts['bodies']:
                pc.set_facecolor('#3182CE')
                pc.set_edgecolor('#1E40AF')
                pc.set_alpha(0.7)
            
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                violin_parts[partname].set_edgecolor('#1E40AF')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(centrality_values.keys())
            ax.set_ylabel('Distribusi Nilai')
            ax.set_title('Perbandingan Distribusi Nilai Centrality', pad=20)
            plt.tight_layout()
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<div class='sub-header'>Degree Centrality</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        **Degree Centrality** mengukur jumlah koneksi langsung yang dimiliki oleh setiap aktor dalam jaringan.
        Aktor dengan degree centrality tinggi memiliki banyak interaksi langsung dan sering menjadi pusat diskusi.
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi menggunakan Gephi dan Top Aktor
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='gephi-viz'>", unsafe_allow_html=True)
            st.image("images/degree_centrality_gephi.png", caption="Visualisasi Gephi - Degree Centrality Network")
            st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: #64748B;">
                Visualisasi jaringan dengan ukuran node proporsional terhadap nilai degree centrality
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.subheader("Top Aktor - Degree Centrality")
            df_degree = display_top_nodes(centrality_metrics["degree"], "Nilai", 10)
            st.dataframe(df_degree, hide_index=True)
            
            st.markdown("""
            **Interpretasi Degree Centrality:**
            
            Aktor dengan degree centrality tinggi:
            - Memiliki banyak interaksi langsung
            - Sering menjadi pusat diskusi
            - Memiliki pengaruh langsung yang besar
            - Berperan sebagai penyebar informasi utama
            
            Aktor-aktor ini adalah target strategis untuk kolaborasi dalam kampanye program stunting karena dapat menjangkau banyak aktor lain secara langsung.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='sub-header'>Betweenness Centrality</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        **Betweenness Centrality** mengukur seberapa sering suatu aktor berada di jalur terpendek antara dua aktor lainnya.
        Aktor dengan betweenness centrality tinggi berperan sebagai 'jembatan' atau perantara dalam jaringan dan memiliki
        kontrol terhadap aliran informasi.
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi menggunakan Gephi dan Top Aktor
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='gephi-viz'>", unsafe_allow_html=True)
            st.image("images/betweeness_centrality_gephi.png", caption="Visualisasi Gephi - Betweenness Centrality Network")
            st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: #64748B;">
                Visualisasi jaringan dengan ukuran node proporsional terhadap nilai betweenness centrality
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.subheader("Top Aktor - Betweenness Centrality")
            df_betweenness = display_top_nodes(centrality_metrics["betweenness"], "Nilai", 10)
            st.dataframe(df_betweenness, hide_index=True)
            
            st.markdown("""
            **Interpretasi Betweenness Centrality:**
            
            Aktor dengan betweenness centrality tinggi:
            - Menjadi perantara informasi
            - Menghubungkan kelompok-kelompok berbeda
            - Memiliki kontrol terhadap aliran informasi
            - Penting untuk penyebaran program stunting antar komunitas
            
            Aktor-aktor ini penting untuk menjembatani komunitas yang terpisah dan memastikan informasi menyebar ke seluruh jaringan.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class='sub-header'>Closeness Centrality</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        **Closeness Centrality** mengukur seberapa dekat suatu aktor dengan semua aktor lainnya dalam jaringan.
        Aktor dengan closeness centrality tinggi dapat menyebarkan informasi dengan cepat karena memiliki jarak yang pendek
        ke banyak aktor lain.
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi menggunakan Gephi dan Top Aktor
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='gephi-viz'>", unsafe_allow_html=True)
            st.image("images/closeness_centrality_gephi.png", caption="Visualisasi Gephi - Closeness Centrality Network")
            st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: #64748B;">
                Visualisasi jaringan dengan ukuran node proporsional terhadap nilai closeness centrality
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.subheader("Top Aktor - Closeness Centrality")
            df_closeness = display_top_nodes(centrality_metrics["closeness"], "Nilai", 10)
            st.dataframe(df_closeness, hide_index=True)
            
            st.markdown("""
            **Interpretasi Closeness Centrality:**
            
            Aktor dengan closeness centrality tinggi:
            - Dapat menjangkau banyak aktor dengan cepat
            - Efisien dalam penyebaran informasi
            - Memiliki akses ke sumber informasi beragam
            - Strategis untuk kampanye program stunting
            
            Aktor-aktor ini ideal untuk penyebaran pesan yang cepat dan efisien tentang program stunting ke seluruh jaringan.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab5:
        st.markdown("<div class='sub-header'>Eigenvector Centrality</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        **Eigenvector Centrality** mengukur pengaruh suatu aktor berdasarkan pengaruh aktor lain yang terhubung dengannya.
        Aktor dengan eigenvector centrality tinggi memiliki koneksi dengan aktor-aktor berpengaruh lainnya.
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi menggunakan Gephi dan Top Aktor
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='gephi-viz'>", unsafe_allow_html=True)
            st.image("images/eigenvector_centrality_gephi.png", caption="Visualisasi Gephi - Eigenvector Centrality Network")
            st.markdown("""
            <div style="text-align: center; font-size: 0.8rem; color: #64748B;">
                Visualisasi jaringan dengan ukuran node proporsional terhadap nilai eigenvector centrality
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='data-container'>", unsafe_allow_html=True)
            st.subheader("Top Aktor - Eigenvector Centrality")
            df_eigenvector = display_top_nodes(centrality_metrics["eigenvector"], "Nilai", 10)
            st.dataframe(df_eigenvector, hide_index=True)
            
            st.markdown("""
            **Interpretasi Eigenvector Centrality:**
            
            Aktor dengan eigenvector centrality tinggi:
            - Terhubung dengan aktor-aktor penting
            - Memiliki pengaruh tidak langsung yang besar
            - Berperan sebagai opinion leader
            - Strategis untuk kolaborasi dalam program stunting
            
            Aktor-aktor ini memberikan akses ke aktor berpengaruh lainnya, sehingga penting untuk strategi komunikasi terpadu.
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Kesimpulan dan rekomendasi
    st.markdown("<div class='sub-header'>Kesimpulan dan Rekomendasi</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='data-container'>", unsafe_allow_html=True)
        st.subheader("Kesimpulan")
        st.markdown("""
        1. **Struktur Jaringan**: Percakapan tentang program stunting membentuk jaringan dengan beberapa aktor kunci yang berperan sebagai pusat informasi dan penyebar diskusi.
        
        2. **Aktor Berpengaruh**: Terdapat perbedaan aktor berpengaruh berdasarkan metrik centrality yang berbeda, menunjukkan peran yang beragam dalam jaringan.
        
        3. **Pola Komunikasi**: Informasi tentang program stunting menyebar melalui jalur-jalur tertentu dengan aktor perantara yang memiliki peran strategis.
        
        4. **Kelompok Diskusi**: Terbentuk beberapa kelompok diskusi yang saling terhubung melalui aktor-aktor dengan betweenness centrality tinggi.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='data-container'>", unsafe_allow_html=True)
        st.subheader("Rekomendasi")
        st.markdown("""
        1. **Kerjasama dengan Aktor Kunci**: Melibatkan aktor-aktor dengan centrality tinggi dalam kampanye program stunting untuk meningkatkan jangkauan dan dampak.
        
        2. **Strategi Komunikasi Bertarget**: Mengembangkan strategi komunikasi yang berbeda untuk setiap jenis aktor berdasarkan perannya dalam jaringan.
        
        3. **Penguatan Jaringan**: Meningkatkan konektivitas antar kelompok diskusi untuk memastikan informasi program stunting menyebar secara merata.
        
        4. **Monitoring Percakapan**: Melakukan pemantauan berkelanjutan terhadap percakapan program stunting di media sosial untuk mengidentifikasi perubahan pola dan aktor berpengaruh.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Metodologi
    st.markdown("<div class='sub-header'>Metodologi</div>", unsafe_allow_html=True)
    st.markdown("<div class='data-container'>", unsafe_allow_html=True)
    st.markdown("""
    **Analisis Jaringan Sosial (Social Network Analysis)** dilakukan dengan langkah-langkah berikut:
    
    1. **Pengumpulan Data**: Data percakapan tentang program stunting dikumpulkan dari media sosial X.
    
    2. **Pemodelan Jaringan**: Interaksi antar pengguna dimodelkan sebagai directed graph, dengan username sebagai nodes dan reply sebagai edges.
    
    3. **Analisis Centrality**: Berbagai metrik centrality (degree, betweenness, closeness, eigenvector) dihitung untuk mengidentifikasi aktor-aktor kunci.
    
    4. **Visualisasi**: Jaringan divisualisasikan menggunakan software Gephi untuk mendapatkan representasi visual yang optimal, dengan ukuran node proporsional terhadap nilai centrality.
    
    5. **Interpretasi**: Hasil analisis diinterpretasikan untuk mengidentifikasi peran dan posisi aktor dalam jaringan serta pola aliran informasi.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Notes tentang penggunaan Gephi
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    **Catatan tentang Visualisasi Gephi**:
    
    Visualisasi jaringan pada aplikasi ini menggunakan hasil ekspor dari software Gephi, yang memungkinkan:
    
    1. Representasi visual yang lebih optimal dan estetis dibandingkan visualisasi yang dihasilkan secara langsung
    2. Penggunaan algoritma layout yang lebih canggih untuk menampilkan struktur jaringan dengan lebih jelas
    3. Pengaturan warna dan ukuran node yang lebih presisi berdasarkan nilai centrality
    
    Data centrality metrics tetap dihitung secara real-time menggunakan NetworkX untuk memberikan informasi akurat tentang aktor-aktor berpengaruh.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; color: #64748B; font-size: 0.8rem;">
        Analisis Jaringan Sosial Program Stunting &copy; 2025
    </div>
    """, unsafe_allow_html=True)
    
else:
    st.warning("Data tidak tersedia. Pastikan file data telah tersedia di lokasi yang benar.")