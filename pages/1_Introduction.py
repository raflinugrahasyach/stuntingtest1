import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Introduction - Analisis Sentimen Program Stunting",
    page_icon="📊",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .content-text {
        font-size: 18px;
        text-align: justify;
    }
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .caption-text {
        font-size: 14px;
        text-align: center;
        font-style: italic;
    }
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Judul halaman
st.markdown("<div class='main-header'>📋 Introduction</div>", unsafe_allow_html=True)

# Konten utama
st.markdown("""
<div class='content-text'>
<h2>Analisis Sentimen dan Jaringan Sosial pada Media Sosial X untuk Menilai Persepsi Publik terhadap Program Stunting di Indonesia</h2>
</div>
""", unsafe_allow_html=True)

# Tampilkan konten dalam tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Latar Belakang & Tujuan", 
    "Manfaat Penelitian", 
    "Batasan & Penelitian Terdahulu", 
    "Metodologi", 
    "Data Sampel"
])

with tab1:
    # LATAR BELAKANG 1 - MASALAH KESEHATAN
    st.markdown("<div class='section-header'>Latar Belakang</div>", unsafe_allow_html=True)
    
    # Placeholder untuk gambar pertama (Latar Belakang 1)
    st.markdown("<p class='content-text'><strong>Stunting tetap menjadi masalah kesehatan yang memengaruhi kualitas SDM masa depan.</strong></p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("images/latbel1.png", 
                caption="Latar Belakang: Stunting sebagai masalah kesehatan yang memengaruhi kualitas SDM", 
                use_container_width=True)
    
    st.markdown("""
    <div class='content-text'>
    <p>Menurut WHO, pada 2018, stunting mempengaruhi 149 juta (21,9%) anak di bawah 5 tahun, dengan kekurangan gizi menyebabkannya 45% kematian anak.</p>
    
    <p>Kekurangan gizi yang menyebabkan stunting (Zaleha & Idris, 2022).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # LATAR BELAKANG 2 - TARGET PEMERINTAH
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<p class='content-text'><strong>Pemerintah menargetkan penurunan stunting dari 21.6% pada 2022 menjadi 14% pada 2024 (Muhaimin et al., 2023).</strong></p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("images/latbel2.png", 
                caption="Latar Belakang: Target penurunan stunting oleh pemerintah", 
                use_container_width=True)
    
    # LATAR BELAKANG 3 - PERAN TWITTER
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("<p class='content-text'><strong>Twitter berperan penting dalam membentuk opini publik tentang kesehatan dan stunting (Inayah & Purba, 2020).</strong></p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("images/latbel3.png", 
                caption="Latar Belakang: Peran Twitter dalam membentuk opini publik", 
                use_container_width=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # TUJUAN PENELITIAN
    st.markdown("<div class='section-header'>Tujuan Penelitian</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-text'>
    <ol>
        <li>Mengidentifikasi dan mengklasifikasikan sentimen publik terhadap program stunting tahun 2024 di media sosial Twitter menggunakan VADER Lexicon dan BERT.</li>
        <br>
        <li>Membandingkan algoritma machine learning seperti Gradient-Boosted Tree (GBT), Support Vector Machine (SVM), dan Extreme Gradient Boosting (XGBoost) untuk menentukan model dengan akurasi tertinggi dalam klasifikasi sentimen publik terkait program stunting.</li>
        <br>
        <li>Menganalisis kata-kata yang paling sering muncul di kalangan pengguna media sosial Twitter terkait program stunting di Indonesia.</li>
        <br>
        <li>Menentukan aktor-aktor kunci yang berperan penting dalam jaringan sosial yang terbentuk dari percakapan terkait program stunting di Twitter melalui analisis jaringan sosial (SNA).</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-header'>Manfaat Penelitian</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-text'>
    <ol>
        <li>Memberikan wawasan tentang persepsi/sentimen publik.</li>
        <br>
        <li>Mengidentifikasi aktor kunci dalam pembentukan opini publik.</li>
        <br>
        <li>Memberikan masukan kepada pemerintah.</li>
        <br>
        <li>Memberikan kontribusi pada pengembangan ilmu data.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>Batasan Masalah</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='content-text'>
        <p>Penelitian ini terbatas pada analisis data dari media sosial X terkait persepsi publik terhadap program stunting di tahun 2024. Fokus utama adalah pada:</p>
        
        <ul>
            <li>Data dikumpulkan dari Juli - Oktober 2024</li>
            <li>Kata kunci "Stunting" & "Program Stunting"</li>
            <li>Tweet berbahasa Indonesia</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='sub-header'>Penelitian Terdahulu</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='content-text'>
        <p>Beberapa studi terkait analisis sentimen dan jaringan sosial telah dilakukan untuk menilai persepsi publik terhadap isu kesehatan:</p>
        
        <ul>
            <li>Analisis Akun Twitter Berpengaruh terkait Covid-19 menggunakan Social Network Analysis (Kartino et al, 2021)</li>
            <li>Analisis sentimen terhadap pelayanan Kesehatan berdasarkan ulasan Google Maps menggunakan BERT (Widagdo et al, 2023)</li>
            <li>Social Media Analysis and Topic Modeling: Case Study of Stunting in Indonesia (Muhaimin et al, 2023)</li>
            <li>Tag me a label with multi-arm: Active learning for telugu sentiment analysis (Mukku et al, 2017)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
with tab4:
    st.markdown("<div class='section-header'>Metodologi Penelitian</div>", unsafe_allow_html=True)
    
    # METODOLOGI - PENGUMPULAN DATA
    st.markdown("<div class='sub-header'>Pengumpulan Data</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='content-text'>
        <p>Data yang digunakan dalam penelitian ini berasal dari media sosial X, yang dipilih karena banyaknya percakapan yang terjadi seputar program stunting. Pengumpulan data dilakukan dengan cara scraping tweet terkait stunting menggunakan API dari media sosial X.</p>
        
        <h4>Proses Pengumpulan Data:</h4>
        <ul>
            <li>Alat yang Digunakan: Tweet Harvest</li>
            <li>Periode Pengumpulan Data: Juli - Oktober 2024</li>
            <li>Fokus: Percakapan terkait program stunting</li>
            <li>Kolom yang digunakan untuk Analisis:
                <ul>
                    <li>full_text</li>
                    <li>username</li>
                    <li>reply_count</li>
                </ul>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.image("images/flow1.png", 
                 caption="Flowchart Metodologi Penelitian", 
                 use_container_width=True)
    
    # FLOWCHART ANALISIS SENTIMEN
    st.markdown("<div class='sub-header'>Flowchart Analisis Sentimen</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("images/flow3.png", 
                 caption="Flowchart Analisis Sentimen", 
                 use_container_width=True)
    
    # TEXT PREPROCESSING & LABELING
    st.markdown("<div class='sub-header'>Flowchart Text Preprocessing & Labeling Data</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("images/flow2.png", 
                 caption="Metodologi: Text Preprocessing & Labeling Data", 
                 use_container_width=True)
    
    # ANALISIS SENTIMEN & JARINGAN SOSIAL
    st.markdown("<div class='sub-header'>Flowchart Klasifikasi Sentimen & Analisis Jaringan Sosial</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("images/flow4.png", 
                 caption="Metodologi: Flowchart Klasifikasi Sentimen", 
                 use_container_width=True)
    
    with col2:
        st.image("images/flow5.png", 
                 caption="Metodologi: Flowchart Analisis Jaringan Sosial", 
                 use_container_width=True)
    
with tab5:
    st.markdown("<div class='section-header'>Data Sampel</div>", unsafe_allow_html=True)
    
    try:
        # Membaca data aktual (jika ada)
        df = pd.read_csv('data/db_merge_with_sentiment_24apr.csv')
        
        # Menampilkan 5 data teratas
        st.dataframe(df.head(5), use_container_width=True)
        
        # Visualisasi distribusi sentimen dari data sampel
        st.markdown("<div class='sub-header'>Distribusi Sentimen dari Data</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Jika BERT_Label ada dalam dataset
            if 'BERT_Label' in df.columns:
                bert_counts = df['BERT_Label'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
                ax.set_title('Distribusi Sentimen BERT')
                ax.axis('equal')  
                
                st.pyplot(fig)
            else:
                st.warning("Kolom BERT_Label tidak ditemukan dalam dataset")
        
        with col2:
            # Jika VADER_Label ada dalam dataset
            if 'VADER_Label' in df.columns:
                vader_counts = df['VADER_Label'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('magma'))
                ax.set_title('Distribusi Sentimen VADER')
                ax.axis('equal')
                
                st.pyplot(fig)
            else:
                st.warning("Kolom VADER_Label tidak ditemukan dalam dataset")
    
    except FileNotFoundError:
        st.error("File data/db_merge_with_sentiment_24apr.csv tidak ditemukan.")
        
        # Menggunakan data sampel sebagai fallback
        st.markdown("<div class='sub-header'>Menggunakan data sampel:</div>", unsafe_allow_html=True)
        
        data = {
            'username': ['oravine_', 'sebayu_94FM', 'raquellstep', 'ajengrizkyJ'],
            'full_text': [
                'cuma yang nggak mengalami PTSD yang digunain buat perhitungan IQ. Ini salah satu contoh yang nunjukin sampel yang digunain di sini berasal dari wilayah tertentu dengan kondisi sosial tertentu termasuk disebutin daerah yang terdampak defisiensi yodium cacingan atau stunting.',
                'Dinas Pekerjaan Umum dan Penataan Ruang (DPUPR) Kota Tegal bersama Kelurahan Tegalsari serta Puskesmas Tegal Barat bersinergi dan berkolaborasi melalui inovasinya dalam menangani kasus stunting. Selengkapnya di https://t.co/VBeJ8wCaE2 #infotegal #kotategal #pemkottegal https://t.co/zgVjt1Hc78',
                '@ajengrizkyJ @bibahjenner @tanyakanrl kak maaf bgt ya kalo salah tp yg aku sebut jg bukan ibunya yg stunting tp nnt si anaknya. itu jg faktor tidak langsung. makanya kan aku blg cmiiw',
                '@raquellstep @bibahjenner @tanyakanrl Belajar yg bener besok balik lagi.. stunting itu karena gizi anak sejak kehamilan sampe usia tertentu tidak terpenuhi.. bukan masalah ibunya tidak tinggi.. kocak banget ortunya pendek disebut stunting..'
            ],
            'BERT_Label': ['neutral', 'neutral', 'negative', 'neutral'],
            'VADER_Label': ['neutral', 'neutral', 'negative', 'neutral']
        }
        
        df_sample = pd.DataFrame(data)
        
        # Tampilkan data sampel
        st.dataframe(df_sample, use_container_width=True)
        
        # Visualisasi distribusi sentimen dari data sampel
        st.markdown("<div class='sub-header'>Distribusi Sentimen dari Data Sampel</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            bert_counts = df_sample['BERT_Label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
            ax.set_title('Distribusi Sentimen BERT')
            ax.axis('equal')
            
            st.pyplot(fig)
        
        with col2:
            vader_counts = df_sample['VADER_Label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('magma'))
            ax.set_title('Distribusi Sentimen VADER')
            ax.axis('equal')
            
            st.pyplot(fig)

# Tambahkan informasi mengenai pengembangan aplikasi streamlit
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class='content-text'>
<h3>Tentang Halaman Introduction</h3>

<p>Halaman ini memberikan gambaran umum tentang penelitian analisis sentimen terhadap program stunting di Indonesia.
Navigasikan ke halaman selanjutnya untuk melihat hasil analisis berdasarkan rumusan masalah.</p>
</div>
""", unsafe_allow_html=True)