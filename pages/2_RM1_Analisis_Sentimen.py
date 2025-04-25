import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import ast

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Program Stunting",
    page_icon="📊",
    layout="wide"
)

# Fungsi untuk load data (ganti dengan path yang sesuai)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/db_merge_with_sentiment_24apr.csv')
        # Mengubah string representasi list menjadi list yang sebenarnya
        df['Stemmed_Text'] = df['Stemmed_Text'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Judul halaman
st.title("Analisis Sentimen Publik terhadap Program Stunting 2024")
st.subheader("Menggunakan VADER Lexicon dan BERT")

# Deskripsi analisis
st.markdown("""
    Halaman ini menampilkan hasil analisis sentimen terhadap percakapan di media sosial X terkait program stunting 
    di Indonesia. Analisis dilakukan menggunakan dua pendekatan:
    
    1. **BERT** (Bidirectional Encoder Representations from Transformers) - Model deep learning berbasis transformer
    2. **VADER** (Valence Aware Dictionary and sEntiment Reasoner) - Lexicon dan rule-based sentiment analysis tool
    
    Perbandingan hasil dari kedua metode ini memberikan pemahaman yang lebih komprehensif mengenai sentimen publik.
""")

# Load data
df = load_data()

# Cek apakah data berhasil dimuat
if df.empty:
    st.warning("Data tidak tersedia. Silakan periksa file sumber data.")
else:
    # Tab untuk berbagai visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Sentimen", "WordCloud", "Kata Teratas", "Data Sampel"])
    
    with tab1:
        st.header("Distribusi Sentimen")
        
        # Membuat layout 2 kolom
        col1, col2 = st.columns(2)
        
        with col1:
            # Hitung distribusi sentimen BERT
            bert_counts = df['BERT_Label'].value_counts().reset_index()
            bert_counts.columns = ['Sentimen', 'Jumlah']
            
            # Buat pie chart untuk BERT
            fig_bert = px.pie(
                bert_counts, 
                values='Jumlah', 
                names='Sentimen',
                title='Distribusi Sentimen BERT',
                color='Sentimen',
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                },
                hole=0.4
            )
            fig_bert.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_bert, use_container_width=True)
            
            # Tampilkan statistik BERT
            st.metric("Total Data BERT", len(df))
            st.write(bert_counts)
        
        with col2:
            # Hitung distribusi sentimen VADER
            vader_counts = df['VADER_Label'].value_counts().reset_index()
            vader_counts.columns = ['Sentimen', 'Jumlah']
            
            # Buat pie chart untuk VADER
            fig_vader = px.pie(
                vader_counts, 
                values='Jumlah', 
                names='Sentimen',
                title='Distribusi Sentimen VADER',
                color='Sentimen',
                color_discrete_map={
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#F44336'
                },
                hole=0.4
            )
            fig_vader.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_vader, use_container_width=True)
            
            # Tampilkan statistik VADER
            st.metric("Total Data VADER", len(df))
            st.write(vader_counts)
        
        # Perbandingan distribusi dalam bar chart
        st.subheader("Perbandingan Distribusi Sentimen BERT vs VADER")
        
        # Siapkan data untuk bar chart
        bert_dict = dict(zip(bert_counts['Sentimen'], bert_counts['Jumlah']))
        vader_dict = dict(zip(vader_counts['Sentimen'], vader_counts['Jumlah']))
        
        labels = sorted(list(set(list(bert_dict.keys()) + list(vader_dict.keys()))))
        bert_values = [bert_dict.get(label, 0) for label in labels]
        vader_values = [vader_dict.get(label, 0) for label in labels]
        
        # Buat bar chart perbandingan
        fig_comparison = go.Figure(data=[
            go.Bar(name='BERT', x=labels, y=bert_values, marker_color='#1E88E5'),
            go.Bar(name='VADER', x=labels, y=vader_values, marker_color='#FFA000')
        ])
        fig_comparison.update_layout(
            barmode='group',
            xaxis_title='Sentimen',
            yaxis_title='Jumlah',
            legend_title='Metode'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab2:
        st.header("WordCloud Berdasarkan Sentimen")
        
        # Fungsi untuk membersihkan teks
        def clean_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            text = re.sub(r"['']", '', str(text))
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)
            return text
        
        # Dropdown untuk memilih metode analisis sentimen
        sentiment_method = st.radio(
            "Pilih Metode Analisis Sentimen:",
            ["BERT", "VADER"],
            horizontal=True
        )
        
        label_column = 'BERT_Label' if sentiment_method == 'BERT' else 'VADER_Label'
        
        # Buat layout 3 kolom untuk wordcloud
        wcol1, wcol2, wcol3 = st.columns(3)
        
        # Proses wordcloud untuk masing-masing sentimen
        for sentiment, col in zip(['positive', 'neutral', 'negative'], [wcol1, wcol2, wcol3]):
            # Filter data berdasarkan sentimen
            filtered_data = df[df[label_column] == sentiment]['Stemmed_Text']
            
            if len(filtered_data) > 0:
                # Gabungkan semua teks
                combined_text = ' '.join(filtered_data.apply(clean_text))
                
                # Buat wordcloud
                if combined_text.strip():
                    wordcloud = WordCloud(
                        width=600, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis'
                    ).generate(combined_text)
                    
                    # Tampilkan di kolom yang sesuai
                    with col:
                        st.subheader(f"Sentimen {sentiment.capitalize()}")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        st.write(f"Jumlah data: {len(filtered_data)}")
                else:
                    with col:
                        st.warning(f"Tidak ada data untuk sentimen {sentiment}")
            else:
                with col:
                    st.warning(f"Tidak ada data untuk sentimen {sentiment}")
    
    with tab3:
        st.header("10 Kata Teratas Berdasarkan Sentimen")
        
        # Dropdown untuk memilih metode analisis sentimen
        sentiment_method_tab3 = st.radio(
            "Pilih Metode Analisis Sentimen:",
            ["BERT", "VADER"],
            horizontal=True,
            key="sentiment_method_tab3"
        )
        
        label_column = 'BERT_Label' if sentiment_method_tab3 == 'BERT' else 'VADER_Label'
        
        # Function to get top-N tokens per label
        def top_tokens(df, label_col, n=10):
            topn = {}
            for label in df[label_col].unique():
                # Get all tokens for this label, flattening the lists
                all_tokens = []
                for tokens in df[df[label_col] == label]['Stemmed_Text']:
                    if isinstance(tokens, list):
                        all_tokens.extend(tokens)
                    elif isinstance(tokens, str) and tokens.startswith('['):
                        try:
                            # Try to parse as list
                            parsed = ast.literal_eval(tokens)
                            if isinstance(parsed, list):
                                all_tokens.extend(parsed)
                        except:
                            # If parsing fails, split by spaces
                            all_tokens.extend(tokens.split())
                    else:
                        # If it's just a string, split by spaces
                        all_tokens.extend(str(tokens).split())
                
                # Count occurrences and get top n
                topn[label] = Counter(all_tokens).most_common(n)
            return topn
        
        # Calculate top tokens
        top_n = st.slider("Pilih jumlah kata teratas:", min_value=5, max_value=20, value=10)
        top_tokens_data = top_tokens(df, label_column, n=top_n)
        
        # Buat layout untuk menampilkan visualisasi
        vis_col1, vis_col2, vis_col3 = st.columns(3)
        
        # Untuk setiap sentimen, tampilkan bar chart kata teratas
        for sentiment, col in zip(['positive', 'neutral', 'negative'], [vis_col1, vis_col2, vis_col3]):
            if sentiment in top_tokens_data:
                # Extract tokens and counts
                tokens, counts = zip(*top_tokens_data[sentiment]) if top_tokens_data[sentiment] else ([], [])
                
                if tokens:
                    # Create bar chart
                    with col:
                        st.subheader(f"Sentimen {sentiment.capitalize()}")
                        
                        # Create a dataframe for plotting
                        top_df = pd.DataFrame({
                            'Token': tokens,
                            'Frekuensi': counts
                        })
                        
                        # Horizontal bar chart with Plotly
                        fig = px.bar(
                            top_df,
                            x='Frekuensi',
                            y='Token',
                            orientation='h',
                            title=f"Top {top_n} Kata - {sentiment.capitalize()}",
                            color='Frekuensi',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col:
                        st.warning(f"Tidak ada data untuk sentimen {sentiment}")
            else:
                with col:
                    st.warning(f"Tidak ada data untuk sentimen {sentiment}")
    
    with tab4:
        st.header("Sampel Data dengan Label Sentimen")
        
        # Dropdown untuk filter berdasarkan label
        filter_options = ['Semua'] + sorted(df['BERT_Label'].unique().tolist())
        selected_filter = st.selectbox('Filter berdasarkan sentimen BERT:', filter_options)
        
        # Filter data berdasarkan pilihan
        if selected_filter != 'Semua':
            filtered_df = df[df['BERT_Label'] == selected_filter]
        else:
            filtered_df = df
        
        # Menentukan kolom yang akan ditampilkan
        display_columns = ['username', 'full_text', 'BERT_Label', 'VADER_Label']
        
        # Menampilkan data
        st.dataframe(
            filtered_df[display_columns],
            column_config={
                "username": "Username",
                "full_text": st.column_config.TextColumn("Tweet", width="medium"),
                "BERT_Label": "Label BERT",
                "VADER_Label": "Label VADER"
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Menampilkan ringkasan statistik
        st.subheader("Ringkasan Data")
        
        # Perbandingan label BERT dan VADER
        comparison_df = pd.crosstab(df['BERT_Label'], df['VADER_Label'])
        st.write("Perbandingan Label BERT dan VADER:")
        st.write(comparison_df)
        
        # Visualisasi heatmap perbandingan
        fig = px.imshow(
            comparison_df, 
            text_auto=True, 
            color_continuous_scale='viridis',
            title="Matriks Perbandingan Label BERT dan VADER"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Analisis Sentimen dan Jaringan Sosial pada Media Sosial X untuk Menilai Persepsi Publik terhadap Program Stunting di Indonesia")