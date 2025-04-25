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

# Judul halaman
st.title("📋 Introduction")

# Konten utama
st.markdown("""
## Analisis Sentimen dan Jaringan Sosial pada Media Sosial X untuk Menilai Persepsi Publik terhadap Program Stunting di Indonesia
""")

# Tampilkan konten dalam tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Latar Belakang & Tujuan", 
    "Manfaat Penelitian", 
    "Batasan & Penelitian Terdahulu", 
    "Metodologi", 
    "Data Sampel"
])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Latar Belakang
        
        Stunting tetap menjadi masalah kesehatan yang memengaruhi kualitas SDM masa depan. 
        
        Menurut WHO, pada 2018, stunting mempengaruhi 149 juta (21,9%) anak di bawah 5 tahun, dengan kekurangan gizi menyebabkannya 45% kematian anak.
        
        Kekurangan gizi yang menyebabkan stunting (Zaleha & Idris, 2022).
        
        Pemerintah menargetkan penurunan stunting dari 21.6% pada 2022 menjadi 14% pada 2024 (Muhaimin et al., 2023).
        
        Twitter berperan penting dalam membentuk opini publik tentang kesehatan dan stunting (Inayah & Purba, 2020).
        """)
    
    with col2:
        # Gunakan placeholder image karena tidak ada gambar dari PPT yang disediakan
        # Di implementasi nyata, kita akan mengganti ini dengan gambar aktual dari PPT
        st.image("https://via.placeholder.com/400x300?text=Ilustrasi+Stunting", caption="Ilustrasi Program Stunting di Indonesia", use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### Tujuan Penelitian
    
    1. Mengidentifikasi dan mengklasifikasikan sentimen publik terhadap program stunting tahun 2024 di media sosial Twitter menggunakan VADER Lexicon dan BERT.
    
    2. Membandingkan algoritma machine learning seperti Gradient-Boosted Tree (GBT), Support Vector Machine (SVM), dan Extreme Gradient Boosting (XGBoost) untuk menentukan model dengan akurasi tertinggi dalam klasifikasi sentimen publik terkait program stunting.
    
    3. Menganalisis kata-kata yang paling sering muncul di kalangan pengguna media sosial Twitter terkait program stunting di Indonesia.
                
    4. Menentukan aktor-aktor kunci yang berperan penting dalam jaringan sosial yang terbentuk dari percakapan terkait program stunting di Twitter melalui analisis jaringan sosial (SNA).
    """)

with tab2:
    st.markdown("""
    ### Manfaat Penelitian
    
    1. Memberikan wawasan tentang persepsi/sentimen publik.
    
    2. Mengidentifikasi aktor kunci dalam pembentukan opini publik.
    
    3. Memberikan masukan kepada pemerintah.
    
    4. Memberikan kontribusi pada pengembangan ilmu data.
    """)
    
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Batasan Masalah
        
        Penelitian ini terbatas pada analisis data dari media sosial X terkait persepsi publik terhadap program stunting di tahun 2024. Fokus utama adalah pada:
        
        - Data dikumpulkan dari Juli - Oktober 2024
        - Kata kunci "Stunting" & "Program Stunting"
        - Tweet berbahasa Indonesia
        """)
    
    with col2:
        st.markdown("""
        ### Penelitian Terdahulu
        
        Beberapa studi terkait analisis sentimen dan jaringan sosial telah dilakukan untuk menilai persepsi publik terhadap isu kesehatan:
        
        - Analisis Akun Twitter Berpengaruh terkait Covid-19 menggunakan Social Network Analysis (Kartino et al, 2021)
        - Analisis sentimen terhadap pelayanan Kesehatan berdasarkan ulasan Google Maps menggunakan BERT (Widagdo et al, 2023)
        - Social Media Analysis and Topic Modeling: Case Study of Stunting in Indonesia (Muhaimin et al, 2023)
        - Tag me a label with multi-arm: Active learning for telugu sentiment analysis (Mukku et al, 2017)
        """)
    
with tab4:
    st.markdown("""
    ### Metodologi Penelitian - Pengumpulan Data
    
    Data yang digunakan dalam penelitian ini berasal dari media sosial X, yang dipilih karena banyaknya percakapan yang terjadi seputar program stunting. Pengumpulan data dilakukan dengan cara scraping tweet terkait stunting menggunakan API dari media sosial X.
    
    #### Proses Pengumpulan Data:
    
    1. Alat yang Digunakan: Tweet Harvest
    2. Periode Pengumpulan Data: Juli - Oktober 2024
    3. Fokus: Percakapan terkait program stunting
    4. Kolom yang digunakan untuk Analisis:
            - full_text
            - username
            - reply_count
    """)
    
with tab5:
    st.markdown("### Data Sampel")
    
    try:
        # Membaca data aktual (bukan dummy)
        df = pd.read_csv('data/db_merge_with_sentiment_24apr.csv')
        
        # Menampilkan 5 data teratas
        st.dataframe(df.head(5), use_container_width=True)
        
        # Visualisasi distribusi sentimen dari data sampel
        st.markdown("#### Distribusi Sentimen dari Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Jika BERT_Label ada dalam dataset
            if 'BERT_Label' in df.columns:
                bert_counts = df['BERT_Label'].value_counts()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
                ax.set_title('Distribusi Sentimen BERT')
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
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
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
                st.pyplot(fig)
            else:
                st.warning("Kolom VADER_Label tidak ditemukan dalam dataset")
    
    except FileNotFoundError:
        st.error("File data/db_merge_with_sentiment_24apr.csv tidak ditemukan.")
        
        # Sebagai fallback, gunakan data dari dokumen yang diberikan
        st.markdown("#### Menggunakan data sampel dari dokumen:")
        
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
        st.markdown("#### Distribusi Sentimen dari Data Sampel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bert_counts = df_sample['BERT_Label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
            ax.set_title('Distribusi Sentimen BERT')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            st.pyplot(fig)
        
        with col2:
            vader_counts = df_sample['VADER_Label'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('magma'))
            ax.set_title('Distribusi Sentimen VADER')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            st.pyplot(fig)

# Tambahkan informasi mengenai pengembangan aplikasi streamlit
st.markdown("---")
st.markdown("""
### Tentang Halaman Introduction

Halaman ini memberikan gambaran umum tentang penelitian analisis sentimen terhadap program stunting di Indonesia.
Navigasikan ke halaman selanjutnya untuk melihat hasil analisis berdasarkan rumusan masalah.
""")