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

# Mengatur tampilan sidebar untuk confusion matrix
show_confusion_matrix = st.sidebar.checkbox("Tampilkan Confusion Matrix", False)

if show_confusion_matrix:
    selected_algorithm = st.sidebar.selectbox(
        "Pilih Algoritma untuk Confusion Matrix:", 
        ["GBDT", "SVC", "XGBoost"]
    )

model_type = st.sidebar.radio(
    "Pilih Model Sentimen:",
    ["BERT", "VADER Lexicon", "Keduanya"]
)

# Tab untuk menampilkan berbagai aspek analisis
tab1, tab2, tab3 = st.tabs(["📊 Perbandingan Metrik", "🔍 Detail Algoritma", "📝 Interpretasi"])

with tab1:
    st.header("Perbandingan Metrik Evaluasi")
    
    if not show_confusion_matrix:
        # Tampilkan keempat metrik secara bersamaan dalam subplot
        if model_type == "Keduanya":
            # Buat subplot 2x2 untuk keempat metrik
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"),
                vertical_spacing=0.15,
                horizontal_spacing=0.08
            )
            
            # Accuracy subplot (baris 1, kolom 1)
            fig.add_trace(go.Bar(
                x=models,
                y=bert_accuracies,
                name='BERT',
                marker_color='indianred',
                text=[f"{acc:.2f}" for acc in bert_accuracies],
                textposition='auto',
                showlegend=True
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=models,
                y=vader_accuracies,
                name='VADER Lexicon',
                marker_color='royalblue',
                text=[f"{acc:.2f}" for acc in vader_accuracies],
                textposition='auto',
                showlegend=True
            ), row=1, col=1)
            
            # Precision subplot (baris 1, kolom 2)
            bert_macro_precisions = [bert_precisions[model]['macro'] for model in models]
            vader_macro_precisions = [vader_precisions[model]['macro'] for model in models]
            
            fig.add_trace(go.Bar(
                x=models,
                y=bert_macro_precisions,
                name='BERT',
                marker_color='indianred',
                text=[f"{prec:.2f}" for prec in bert_macro_precisions],
                textposition='auto',
                showlegend=False
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                x=models,
                y=vader_macro_precisions,
                name='VADER Lexicon',
                marker_color='royalblue',
                text=[f"{prec:.2f}" for prec in vader_macro_precisions],
                textposition='auto',
                showlegend=False
            ), row=1, col=2)
            
            # Recall subplot (baris 2, kolom 1)
            bert_macro_recalls = [bert_recalls[model]['macro'] for model in models]
            vader_macro_recalls = [vader_recalls[model]['macro'] for model in models]
            
            fig.add_trace(go.Bar(
                x=models,
                y=bert_macro_recalls,
                name='BERT',
                marker_color='indianred',
                text=[f"{rec:.2f}" for rec in bert_macro_recalls],
                textposition='auto',
                showlegend=False
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                x=models,
                y=vader_macro_recalls,
                name='VADER Lexicon',
                marker_color='royalblue',
                text=[f"{rec:.2f}" for rec in vader_macro_recalls],
                textposition='auto',
                showlegend=False
            ), row=2, col=1)
            
            # F1 Score subplot (baris 2, kolom 2)
            bert_macro_f1s = [bert_f1_scores[model]['macro'] for model in models]
            vader_macro_f1s = [vader_f1_scores[model]['macro'] for model in models]
            
            fig.add_trace(go.Bar(
                x=models,
                y=bert_macro_f1s,
                name='BERT',
                marker_color='indianred',
                text=[f"{f1:.2f}" for f1 in bert_macro_f1s],
                textposition='auto',
                showlegend=False
            ), row=2, col=2)
            
            fig.add_trace(go.Bar(
                x=models,
                y=vader_macro_f1s,
                name='VADER Lexicon',
                marker_color='royalblue',
                text=[f"{f1:.2f}" for f1 in vader_macro_f1s],
                textposition='auto',
                showlegend=False
            ), row=2, col=2)
            
        else:  # BERT atau VADER
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 Score (Macro)"),
                vertical_spacing=0.15,
                horizontal_spacing=0.08
            )
            
            if model_type == "BERT":
                color = 'indianred'
                accuracies = bert_accuracies
                macro_precisions = [bert_precisions[model]['macro'] for model in models]
                macro_recalls = [bert_recalls[model]['macro'] for model in models]
                macro_f1s = [bert_f1_scores[model]['macro'] for model in models]
            else:  # VADER
                color = 'royalblue'
                accuracies = vader_accuracies
                macro_precisions = [vader_precisions[model]['macro'] for model in models]
                macro_recalls = [vader_recalls[model]['macro'] for model in models]
                macro_f1s = [vader_f1_scores[model]['macro'] for model in models]
            
            # Accuracy
            fig.add_trace(go.Bar(
                x=models,
                y=accuracies,
                name=model_type,
                marker_color=color,
                text=[f"{acc:.2f}" for acc in accuracies],
                textposition='auto'
            ), row=1, col=1)
            
            # Precision
            fig.add_trace(go.Bar(
                x=models,
                y=macro_precisions,
                name=model_type,
                marker_color=color,
                text=[f"{prec:.2f}" for prec in macro_precisions],
                textposition='auto',
                showlegend=False
            ), row=1, col=2)
            
            # Recall
            fig.add_trace(go.Bar(
                x=models,
                y=macro_recalls,
                name=model_type,
                marker_color=color,
                text=[f"{rec:.2f}" for rec in macro_recalls],
                textposition='auto',
                showlegend=False
            ), row=2, col=1)
            
            # F1 Score
            fig.add_trace(go.Bar(
                x=models,
                y=macro_f1s,
                name=model_type,
                marker_color=color,
                text=[f"{f1:.2f}" for f1 in macro_f1s],
                textposition='auto',
                showlegend=False
            ), row=2, col=2)
        
        # Update layout for all subplots
        fig.update_layout(
            title="Perbandingan Metrik Evaluasi Model Klasifikasi Sentimen",
            barmode='group',
            legend_title="Model Sentimen",
            height=700
        )
        
        # Set y-axis range untuk semua subplot
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_yaxes(range=[0, 1], row=i, col=j)
        
        # Tampilkan plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Menampilkan tabel metrik detail
        st.subheader("Detail Metrik per Kelas")
        
        if model_type == "BERT":
            # Buat DataFrame untuk metrik BERT detail
            bert_detail = pd.DataFrame({
                "Algoritma": models,
                "Accuracy": bert_accuracies,
                "Precision (Neg/0)": [bert_precisions[model]['0'] for model in models],
                "Precision (Neu/1)": [bert_precisions[model]['1'] for model in models],
                "Precision (Pos/2)": [bert_precisions[model]['2'] for model in models],
                "Recall (Neg/0)": [bert_recalls[model]['0'] for model in models],
                "Recall (Neu/1)": [bert_recalls[model]['1'] for model in models],
                "Recall (Pos/2)": [bert_recalls[model]['2'] for model in models],
                "F1 (Neg/0)": [bert_f1_scores[model]['0'] for model in models],
                "F1 (Neu/1)": [bert_f1_scores[model]['1'] for model in models],
                "F1 (Pos/2)": [bert_f1_scores[model]['2'] for model in models],
            })
            
            st.dataframe(bert_detail, use_container_width=True)
            
        elif model_type == "VADER Lexicon":
            # Buat DataFrame untuk metrik VADER detail
            vader_detail = pd.DataFrame({
                "Algoritma": models,
                "Accuracy": vader_accuracies,
                "Precision (Neg/0)": [vader_precisions[model]['0'] for model in models],
                "Precision (Neu/1)": [vader_precisions[model]['1'] for model in models],
                "Precision (Pos/2)": [vader_precisions[model]['2'] for model in models],
                "Recall (Neg/0)": [vader_recalls[model]['0'] for model in models],
                "Recall (Neu/1)": [vader_recalls[model]['1'] for model in models],
                "Recall (Pos/2)": [vader_recalls[model]['2'] for model in models],
                "F1 (Neg/0)": [vader_f1_scores[model]['0'] for model in models],
                "F1 (Neu/1)": [vader_f1_scores[model]['1'] for model in models],
                "F1 (Pos/2)": [vader_f1_scores[model]['2'] for model in models],
            })
            
            st.dataframe(vader_detail, use_container_width=True)
            
        else:  # Keduanya
            st.write("**BERT**")
            bert_detail = pd.DataFrame({
                "Algoritma": models,
                "Accuracy": bert_accuracies,
                "Precision (Neg/0)": [bert_precisions[model]['0'] for model in models],
                "Precision (Neu/1)": [bert_precisions[model]['1'] for model in models],
                "Precision (Pos/2)": [bert_precisions[model]['2'] for model in models],
                "Recall (Neg/0)": [bert_recalls[model]['0'] for model in models],
                "Recall (Neu/1)": [bert_recalls[model]['1'] for model in models],
                "Recall (Pos/2)": [bert_recalls[model]['2'] for model in models],
                "F1 (Neg/0)": [bert_f1_scores[model]['0'] for model in models],
                "F1 (Neu/1)": [bert_f1_scores[model]['1'] for model in models],
                "F1 (Pos/2)": [bert_f1_scores[model]['2'] for model in models],
            })
            
            st.dataframe(bert_detail, use_container_width=True)
            
            st.write("**VADER Lexicon**")
            vader_detail = pd.DataFrame({
                "Algoritma": models,
                "Accuracy": vader_accuracies,
                "Precision (Neg/0)": [vader_precisions[model]['0'] for model in models],
                "Precision (Neu/1)": [vader_precisions[model]['1'] for model in models],
                "Precision (Pos/2)": [vader_precisions[model]['2'] for model in models],
                "Recall (Neg/0)": [vader_recalls[model]['0'] for model in models],
                "Recall (Neu/1)": [vader_recalls[model]['1'] for model in models],
                "Recall (Pos/2)": [vader_recalls[model]['2'] for model in models],
                "F1 (Neg/0)": [vader_f1_scores[model]['0'] for model in models],
                "F1 (Neu/1)": [vader_f1_scores[model]['1'] for model in models],
                "F1 (Pos/2)": [vader_f1_scores[model]['2'] for model in models],
            })
            
            st.dataframe(vader_detail, use_container_width=True)
    
    else:  # Tampilkan Confusion Matrix
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