import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi halaman
st.set_page_config(
    page_title="Perbandingan Algoritma ML - Program Stunting",
    page_icon="📊",
    layout="wide"
)

# Judul halaman
st.title("📈 Perbandingan Algoritma Machine Learning (GBT, SVM, dan XGBoost)")
st.subheader("Rumusan Masalah 2: Klasifikasi Sentimen Program Stunting di Media Sosial X")

# Penjelasan singkat
st.markdown("""
Halaman ini menyajikan hasil perbandingan dari tiga algoritma machine learning dalam klasifikasi sentimen publik
terhadap program stunting di Indonesia berdasarkan data dari media sosial X (Twitter):

1. **Gradient Boosted Decision Tree (GBDT)** - Algoritma yang membangun model prediktif dalam bentuk ensemble dari decision tree
2. **Support Vector Classifier (SVC)** - Algoritma yang mencari hyperplane terbaik untuk memisahkan kelas-kelas
3. **Extreme Gradient Boosting (XGBoost)** - Implementasi efisien dari gradient boosting yang dioptimalkan untuk kinerja

Kita membandingkan performa ketiga algoritma dengan menggunakan dua metode analisis sentimen yang berbeda:
**BERT** dan **VADER Lexicon**.
""")

# Data dari laporan
# Menggunakan data real dari Laporan Analisis
models = ['GBDT', 'SVC', 'XGBoost']

# Metrics from the report
bert_accuracies = [0.7529249827942188, 0.7894012388162422, 0.7660013764624914]
vader_accuracies = [0.9559532002752925, 0.9201651754989677, 0.9525120440467997]

# BERT metrics
bert_precisions = {
    'GBDT': {'0': 0.57, '1': 0.84, '2': 0.83, 'macro': 0.75, 'weighted': 0.78},
    'SVC': {'0': 0.71, '1': 0.83, '2': 0.67, 'macro': 0.74, 'weighted': 0.77},
    'XGBoost': {'0': 0.55, '1': 0.83, '2': 0.67, 'macro': 0.68, 'weighted': 0.74}
}

bert_recalls = {
    'GBDT': {'0': 0.44, '1': 0.95, '2': 0.27, 'macro': 0.55, 'weighted': 0.75},
    'SVC': {'0': 0.63, '1': 0.89, '2': 0.30, 'macro': 0.61, 'weighted': 0.75},
    'XGBoost': {'0': 0.48, '1': 0.93, '2': 0.30, 'macro': 0.57, 'weighted': 0.75}
}

bert_f1_scores = {
    'GBDT': {'0': 0.54, '1': 0.84, '2': 0.41, 'macro': 0.60, 'weighted': 0.73},
    'SVC': {'0': 0.67, '1': 0.86, '2': 0.41, 'macro': 0.65, 'weighted': 0.76},
    'XGBoost': {'0': 0.51, '1': 0.85, '2': 0.42, 'macro': 0.59, 'weighted': 0.74}
}

# VADER metrics
vader_precisions = {
    'GBDT': {'0': 0.91, '1': 0.95, '2': 0.98, 'macro': 0.95, 'weighted': 0.96},
    'SVC': {'0': 1.00, '1': 0.92, '2': 0.98, 'macro': 0.97, 'weighted': 0.93},
    'XGBoost': {'0': 0.91, '1': 0.95, '2': 0.98, 'macro': 0.95, 'weighted': 0.95}
}

vader_recalls = {
    'GBDT': {'0': 0.58, '1': 1.00, '2': 0.76, 'macro': 0.78, 'weighted': 0.96},
    'SVC': {'0': 0.17, '1': 1.00, '2': 0.58, 'macro': 0.58, 'weighted': 0.92},
    'XGBoost': {'0': 0.58, '1': 1.00, '2': 0.73, 'macro': 0.77, 'weighted': 0.95}
}

vader_f1_scores = {
    'GBDT': {'0': 0.71, '1': 0.98, '2': 0.85, 'macro': 0.85, 'weighted': 0.95},
    'SVC': {'0': 0.30, '1': 0.96, '2': 0.73, 'macro': 0.66, 'weighted': 0.90},
    'XGBoost': {'0': 0.71, '1': 0.98, '2': 0.83, 'macro': 0.84, 'weighted': 0.95}
}

# Confusion matrices from the report
conf_matrices = {
    'GBDT_BERT': np.array([[71, 73, 18], [13, 871, 28], [42, 99, 44]]),
    'SVC_BERT': np.array([[102, 48, 12], [34, 816, 62], [8, 125, 52]]),
    'XGBoost_BERT': np.array([[77, 69, 16], [25, 849, 38], [38, 106, 41]]),
    'GBDT_VADER': np.array([[30, 22, 0], [0, 1229, 0], [3, 38, 131]]),
    'SVC_VADER': np.array([[9, 43, 0], [0, 1229, 0], [0, 72, 100]]),
    'XGBoost_VADER': np.array([[30, 22, 0], [0, 1229, 0], [3, 44, 125]])
}

# Class support numbers (from the confusion matrices)
class_support = {
    'BERT': {'0': 162, '1': 912, '2': 185},
    'VADER': {'0': 52, '1': 1229, '2': 172}
}

# Sidebar untuk kontrol
st.sidebar.header("Konfigurasi Tampilan")

# Mengatur tampilan sidebar
metric = st.sidebar.selectbox(
    "Pilih Metrik Evaluasi:", 
    ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"]
)

model_type = st.sidebar.radio(
    "Pilih Model Sentimen:",
    ["BERT", "VADER Lexicon", "Keduanya"]
)

# Memilih algoritma untuk confusion matrix jika dipilih
if metric == "Confusion Matrix":
    selected_algorithm = st.sidebar.selectbox(
        "Pilih Algoritma:", 
        ["GBDT", "SVC", "XGBoost"]
    )
    
# Tab untuk menampilkan berbagai aspek analisis
tab1, tab2, tab3 = st.tabs(["📊 Perbandingan Metrik", "🔍 Detail Algoritma", "📝 Interpretasi"])

with tab1:
    st.header("Perbandingan Metrik Evaluasi")
    
    if metric != "Confusion Matrix":
        # Tampilkan perbandingan metrik dalam bentuk grafik bar
        fig = go.Figure()
        
        if model_type in ["BERT", "Keduanya"]:
            if metric == "Accuracy":
                fig.add_trace(go.Bar(
                    x=models,
                    y=bert_accuracies,
                    name='BERT',
                    marker_color='indianred',
                    text=[f"{acc:.5f}" for acc in bert_accuracies],
                    textposition='auto',
                ))
            elif metric == "Precision":
                bert_macro_precisions = [bert_precisions[model]['macro'] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=bert_macro_precisions,
                    name='BERT',
                    marker_color='indianred',
                    text=[f"{prec:.5f}" for prec in bert_macro_precisions],
                    textposition='auto',
                ))
            elif metric == "Recall":
                bert_macro_recalls = [bert_recalls[model]['macro'] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=bert_macro_recalls,
                    name='BERT',
                    marker_color='indianred',
                    text=[f"{rec:.5f}" for rec in bert_macro_recalls],
                    textposition='auto',
                ))
            elif metric == "F1 Score":
                bert_macro_f1s = [bert_f1_scores[model]['macro'] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=bert_macro_f1s,
                    name='BERT',
                    marker_color='indianred',
                    text=[f"{f1:.5f}" for f1 in bert_macro_f1s],
                    textposition='auto',
                ))
        
        if model_type in ["VADER Lexicon", "Keduanya"]:
            if metric == "Accuracy":
                fig.add_trace(go.Bar(
                    x=models,
                    y=vader_accuracies,
                    name='VADER Lexicon',
                    marker_color='royalblue',
                    text=[f"{acc:.5f}" for acc in vader_accuracies],
                    textposition='auto',
                ))
            elif metric == "Precision":
                vader_macro_precisions = [vader_precisions[model]['macro'] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=vader_macro_precisions,
                    name='VADER Lexicon',
                    marker_color='royalblue',
                    text=[f"{prec:.5f}" for prec in vader_macro_precisions],
                    textposition='auto',
                ))
            elif metric == "Recall":
                vader_macro_recalls = [vader_recalls[model]['macro'] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=vader_macro_recalls,
                    name='VADER Lexicon',
                    marker_color='royalblue',
                    text=[f"{rec:.5f}" for rec in vader_macro_recalls],
                    textposition='auto',
                ))
            elif metric == "F1 Score":
                vader_macro_f1s = [vader_f1_scores[model]['macro'] for model in models]
                fig.add_trace(go.Bar(
                    x=models,
                    y=vader_macro_f1s,
                    name='VADER Lexicon',
                    marker_color='royalblue',
                    text=[f"{f1:.5f}" for f1 in vader_macro_f1s],
                    textposition='auto',
                ))
                
        fig.update_layout(
            title=f"Perbandingan {metric} Model Klasifikasi Sentimen",
            xaxis_title="Algoritma",
            yaxis_title=f"{metric}",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            legend_title="Model Sentimen",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        if metric != "Accuracy":
            st.subheader(f"Detail {metric} per Kelas")
            
            if model_type == "BERT":
                if metric == "Precision":
                    detail_data = {model: {k: v for k, v in bert_precisions[model].items() if k in ['0', '1', '2']} for model in models}
                elif metric == "Recall":
                    detail_data = {model: {k: v for k, v in bert_recalls[model].items() if k in ['0', '1', '2']} for model in models}
                else:  # F1 Score
                    detail_data = {model: {k: v for k, v in bert_f1_scores[model].items() if k in ['0', '1', '2']} for model in models}
                
                df = pd.DataFrame({
                    "Algoritma": models,
                    "Negatif (0)": [detail_data[model]['0'] for model in models],
                    "Netral (1)": [detail_data[model]['1'] for model in models],
                    "Positif (2)": [detail_data[model]['2'] for model in models],
                    "Macro Avg": [detail_data[model]['macro'] for model in models],
                    "Support": [f"{class_support['BERT']['0']}, {class_support['BERT']['1']}, {class_support['BERT']['2']}"] * len(models)
                })
                
                st.dataframe(df, use_container_width=True)
                
            elif model_type == "VADER Lexicon":
                if metric == "Precision":
                    detail_data = {model: {k: v for k, v in vader_precisions[model].items() if k in ['0', '1', '2']} for model in models}
                elif metric == "Recall":
                    detail_data = {model: {k: v for k, v in vader_recalls[model].items() if k in ['0', '1', '2']} for model in models}
                else:  # F1 Score
                    detail_data = {model: {k: v for k, v in vader_f1_scores[model].items() if k in ['0', '1', '2']} for model in models}
                
                df = pd.DataFrame({
                    "Algoritma": models,
                    "Negatif (0)": [detail_data[model]['0'] for model in models],
                    "Netral (1)": [detail_data[model]['1'] for model in models],
                    "Positif (2)": [detail_data[model]['2'] for model in models],
                    "Macro Avg": [detail_data[model]['macro'] for model in models],
                    "Support": [f"{class_support['VADER']['0']}, {class_support['VADER']['1']}, {class_support['VADER']['2']}"] * len(models)
                })
                
                st.dataframe(df, use_container_width=True)
                
            else:  # Keduanya
                st.write("**BERT**")
                if metric == "Precision":
                    detail_data = {model: {k: v for k, v in bert_precisions[model].items() if k in ['0', '1', '2']} for model in models}
                elif metric == "Recall":
                    detail_data = {model: {k: v for k, v in bert_recalls[model].items() if k in ['0', '1', '2']} for model in models}
                else:  # F1 Score
                    detail_data = {model: {k: v for k, v in bert_f1_scores[model].items() if k in ['0', '1', '2']} for model in models}
                
                df_bert = pd.DataFrame({
                    "Algoritma": models,
                    "Negatif (0)": [detail_data[model]['0'] for model in models],
                    "Netral (1)": [detail_data[model]['1'] for model in models],
                    "Positif (2)": [detail_data[model]['2'] for model in models],
                    "Macro Avg": [detail_data[model]['macro'] for model in models],
                    "Support": [f"{class_support['BERT']['0']}, {class_support['BERT']['1']}, {class_support['BERT']['2']}"] * len(models)
                })
                
                st.dataframe(df_bert, use_container_width=True)
                
                st.write("**VADER Lexicon**")
                if metric == "Precision":
                    detail_data = {model: {k: v for k, v in vader_precisions[model].items() if k in ['0', '1', '2']} for model in models}
                elif metric == "Recall":
                    detail_data = {model: {k: v for k, v in vader_recalls[model].items() if k in ['0', '1', '2']} for model in models}
                else:  # F1 Score
                    detail_data = {model: {k: v for k, v in vader_f1_scores[model].items() if k in ['0', '1', '2']} for model in models}
                
                df_vader = pd.DataFrame({
                    "Algoritma": models,
                    "Negatif (0)": [detail_data[model]['0'] for model in models],
                    "Netral (1)": [detail_data[model]['1'] for model in models],
                    "Positif (2)": [detail_data[model]['2'] for model in models],
                    "Macro Avg": [detail_data[model]['macro'] for model in models],
                    "Support": [f"{class_support['VADER']['0']}, {class_support['VADER']['1']}, {class_support['VADER']['2']}"] * len(models)
                })
                
                st.dataframe(df_vader, use_container_width=True)
        
    else:  # Confusion Matrix
        st.subheader(f"Confusion Matrix - {selected_algorithm}")
        
        # 2 kolom untuk menampilkan confusion matrix BERT dan VADER
        col1, col2 = st.columns(2)
        
        with col1:
            if model_type in ["BERT", "Keduanya"]:
                st.subheader("BERT")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_matrices[f"{selected_algorithm}_BERT"], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Negatif', 'Netral', 'Positif'], 
                            yticklabels=['Negatif', 'Netral', 'Positif'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix - {selected_algorithm} (BERT)')
                st.pyplot(fig)
        
        with col2:
            if model_type in ["VADER Lexicon", "Keduanya"]:
                st.subheader("VADER Lexicon")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(conf_matrices[f"{selected_algorithm}_VADER"], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Negatif', 'Netral', 'Positif'], 
                            yticklabels=['Negatif', 'Netral', 'Positif'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix - {selected_algorithm} (VADER)')
                st.pyplot(fig)

with tab2:
    st.header("Detail Algoritma Machine Learning")
    
    # Informasi detail tentang algoritma
    selected_algo = st.selectbox(
        "Pilih algoritma untuk melihat detail:",
        ["Gradient Boosted Decision Tree (GBDT)", "Support Vector Classifier (SVC)", "Extreme Gradient Boosting (XGBoost)"]
    )
    
    if selected_algo == "Gradient Boosted Decision Tree (GBDT)":
        st.subheader("Gradient Boosted Decision Tree (GBDT)")
        st.markdown("""
        **Deskripsi:**
        Gradient Boosting adalah teknik machine learning yang digunakan untuk membuat model prediktif. GBDT secara khusus menggunakan decision tree sebagai learner dasar. 
        Algoritma ini membangun model secara bertahap (iteratif), dengan setiap iterasi mencoba memperbaiki kesalahan dari iterasi sebelumnya.
        
        **Cara Kerja:**
        1. Memulai dengan model sederhana (biasanya satu node tree)
        2. Mengidentifikasi "residual" atau kesalahan dari model
        3. Melatih tree berikutnya untuk memprediksi residual tersebut
        4. Menambahkan tree baru ke model untuk mengurangi kesalahan
        5. Mengulangi proses hingga mencapai jumlah iterasi yang ditentukan
        
        """)
        
        
    elif selected_algo == "Support Vector Classifier (SVC)":
        st.subheader("Support Vector Classifier (SVC)")
        st.markdown("""
        **Deskripsi:**
        Support Vector Classifier (SVC) adalah implementasi dari Support Vector Machine (SVM) untuk klasifikasi. SVC mencari hyperplane terbaik 
        yang memisahkan data dari kelas yang berbeda dengan margin maksimal.
        
        
        **Cara Kerja:**
        1. Mentransformasikan data ke dimensi yang lebih tinggi menggunakan kernel
        2. Menemukan hyperplane optimal yang memaksimalkan margin antara kelas
        3. Mengidentifikasi "support vectors" - titik data yang paling dekat dengan hyperplane
        4. Membuat keputusan klasifikasi berdasarkan posisi relatif terhadap hyperplane
        
        """)
        
        st.markdown("**Durasi pelatihan:** 19 detik (BERT), 14 detik (VADER)")
        
    else:  # XGBoost
        st.subheader("Extreme Gradient Boosting (XGBoost)")
        st.markdown("""
        **Deskripsi:**
        XGBoost adalah implementasi teroptimasi dari algoritma Gradient Boosting. Dikenal karena kecepatan dan performa yang unggul,
        XGBoost telah menjadi salah satu algoritma yang paling populer dalam kompetisi machine learning.
        
        **Cara Kerja:**
        1. Menggunakan prinsip gradient boosting seperti GBDT
        2. Menambahkan regularisasi untuk mengurangi overfitting
        3. Menggunakan struktur tree yang dioptimalkan dan algoritma paralel
        4. Menangani nilai yang hilang secara otomatis
        5. Menggunakan teknik "pruning" untuk menghilangkan split yang tidak signifikan
        
        """)
        
        st.markdown("**Durasi pelatihan:** 28 detik (BERT), 26 detik (VADER)")

with tab3:
    st.header("Interpretasi Hasil")
    
    st.subheader("Ringkasan Perbandingan")
    
    # Membuat tabel ringkasan perbandingan
    summary_df = pd.DataFrame({
        'Algoritma': models,
        'Akurasi BERT': bert_accuracies,
        'Akurasi VADER': vader_accuracies,
        'Precision BERT (macro)': [bert_precisions[model]['macro'] for model in models],
        'Precision VADER (macro)': [vader_precisions[model]['macro'] for model in models],
        'Recall BERT (macro)': [bert_recalls[model]['macro'] for model in models],
        'Recall VADER (macro)': [vader_recalls[model]['macro'] for model in models],
        'F1 Score BERT (macro)': [bert_f1_scores[model]['macro'] for model in models],
        'F1 Score VADER (macro)': [vader_f1_scores[model]['macro'] for model in models]
    })
    
    # Format angka dalam tabel
    for col in summary_df.columns:
        if col != 'Algoritma':
            summary_df[col] = summary_df[col].map(lambda x: f"{x:.5f}")
    
    st.dataframe(summary_df, use_container_width=True)
    
    st.subheader("Analisis Performa Model")
    st.markdown("""
    ### Kesimpulan Utama

    1. **XGBoost dan GBDT Menunjukkan Performa Terbaik dengan VADER**
       - Secara konsisten, GBDT dan XGBoost memberikan nilai akurasi tertinggi (95.60% dan 95.25%) saat menggunakan VADER Lexicon.
       - Untuk metriks F1-score (macro), GBDT mencapai 0.85 dan XGBoost 0.84 dengan VADER, mengungguli SVC.

    2. **SVC Unggul dengan BERT**
       - Dalam konteks BERT, SVC mencapai akurasi tertinggi yakni 78.94%, dibandingkan XGBoost (76.60%) dan GBDT (75.29%).
       - SVC juga menunjukkan F1-score (macro) terbaik yakni 0.65 untuk pelabelan BERT.

    3. **BERT vs VADER Lexicon**
       - Model yang menggunakan fitur VADER secara konsisten menunjukkan performa yang lebih baik dibandingkan model BERT.
       - Gap performa antara keduanya cukup signifikan, dengan peningkatan akurasi 17-20% saat menggunakan VADER.
       - Ini mengindikasikan bahwa VADER lebih sesuai untuk analisis sentimen berbahasa Indonesia dalam konteks program stunting.

    3. **Perbandingan Distribusi Kelas**
       - Data VADER memiliki kecenderungan strong bias terhadap label netral (1229 dari 1453 data), sementara distribusi BERT lebih seimbang.
       - Hal ini menjelaskan mengapa akurasi VADER tinggi, namun F1-score makro lebih rendah untuk beberapa model karena kesulitan mendeteksi kelas minoritas.

    ### Implikasi untuk Analisis Sentimen Program Stunting

    Berdasarkan hasil ini, untuk mengklasifikasikan sentimen publik terhadap program stunting di media sosial X:
    
    - **Rekomendasi Model**: GBDT dengan fitur VADER memberikan hasil optimal untuk fokus pada akurasi keseluruhan.
    - **Pertimbangan Keseimbangan Kelas**: Jika deteksi sentimen negatif/positif (kelas minoritas) penting, SVC dengan BERT lebih rekomen.
    - **Trade-off Kecepatan vs Akurasi**: SVC menawarkan pelatihan tercepat (14 detik) dengan akurasi kompetitif pada VADER.
    - **Pengembangan Lebih Lanjut**: Teknik oversampling atau class weighting dapat diterapkan untuk mengatasi ketidakseimbangan kelas.
    """)