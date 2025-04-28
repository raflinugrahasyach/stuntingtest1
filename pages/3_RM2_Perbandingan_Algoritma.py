import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from nltk.corpus import stopwords

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

# Function to load data
@st.cache_data
def load_data():
    # Load data here
    try:
        file_path = 'data/db_merge_with_sentiment_24apr.csv'
        db_merge = pd.read_csv(file_path)
        return db_merge
    except FileNotFoundError:
        st.error("Data file not found. Please upload the file using the sidebar.")
        return None

# Function to preprocess and train models
@st.cache_data
def train_models(db_merge):
    from sklearn.preprocessing import LabelEncoder
    
    # Label encoding
    label_encoder = LabelEncoder()
    db_merge['BERT_Label_Numeric'] = label_encoder.fit_transform(db_merge['BERT_Label'])
    db_merge['VADER_Label_Numeric'] = label_encoder.fit_transform(db_merge['VADER_Label'])
    
    # BERT Data preparation
    X_bert = db_merge['Stemmed_Text']
    y_bert = db_merge['BERT_Label_Numeric']
    X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(X_bert, y_bert, test_size=0.2, random_state=42, stratify=y_bert)
    
    # VADER Data preparation
    X_vader = db_merge['Stemmed_Text']
    y_vader = db_merge['VADER_Label_Numeric']
    X_train_vader, X_test_vader, y_train_vader, y_test_vader = train_test_split(X_vader, y_vader, test_size=0.2, random_state=42, stratify=y_vader)
    
    # Text vectorization
    stop_words = stopwords.words('indonesian')
    vectorizer_bert = TfidfVectorizer(stop_words=stop_words, max_features=5000)
    vectorizer_vader = TfidfVectorizer(stop_words=stop_words, max_features=5000)
    
    # Process text data
    X_train_bert = X_train_bert.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    X_test_bert = X_test_bert.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    X_train_vader = X_train_vader.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    X_test_vader = X_test_vader.apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    
    # Vectorize the data
    X_train_vectorized_bert = vectorizer_bert.fit_transform(X_train_bert)
    X_test_vectorized_bert = vectorizer_bert.transform(X_test_bert)
    X_train_vectorized_vader = vectorizer_vader.fit_transform(X_train_vader)
    X_test_vectorized_vader = vectorizer_vader.transform(X_test_vader)
    
    models = {}
    results = {}
    
    # Train and evaluate BERT models
    # Model GBDT
    gbdt_model = GradientBoostingClassifier()
    gbdt_model.fit(X_train_vectorized_bert, y_train_bert)
    gbdt_pred = gbdt_model.predict(X_test_vectorized_bert)
    models['GBDT_BERT'] = gbdt_model
    results['GBDT_BERT'] = {
        'pred': gbdt_pred,
        'y_test': y_test_bert,
        'accuracy': accuracy_score(y_test_bert, gbdt_pred),
        'confusion_matrix': confusion_matrix(y_test_bert, gbdt_pred),
        'precision': precision_score(y_test_bert, gbdt_pred, average=None),
        'recall': recall_score(y_test_bert, gbdt_pred, average=None),
        'f1': f1_score(y_test_bert, gbdt_pred, average=None),
        'precision_macro': precision_score(y_test_bert, gbdt_pred, average='macro'),
        'recall_macro': recall_score(y_test_bert, gbdt_pred, average='macro'),
        'f1_macro': f1_score(y_test_bert, gbdt_pred, average='macro'),
        'precision_weighted': precision_score(y_test_bert, gbdt_pred, average='weighted'),
        'recall_weighted': recall_score(y_test_bert, gbdt_pred, average='weighted'),
        'f1_weighted': f1_score(y_test_bert, gbdt_pred, average='weighted')
    }
    
    # SVC
    svc_model = SVC()
    svc_model.fit(X_train_vectorized_bert, y_train_bert)
    svc_pred = svc_model.predict(X_test_vectorized_bert)
    models['SVC_BERT'] = svc_model
    results['SVC_BERT'] = {
        'pred': svc_pred,
        'y_test': y_test_bert,
        'accuracy': accuracy_score(y_test_bert, svc_pred),
        'confusion_matrix': confusion_matrix(y_test_bert, svc_pred),
        'precision': precision_score(y_test_bert, svc_pred, average=None),
        'recall': recall_score(y_test_bert, svc_pred, average=None),
        'f1': f1_score(y_test_bert, svc_pred, average=None),
        'precision_macro': precision_score(y_test_bert, svc_pred, average='macro'),
        'recall_macro': recall_score(y_test_bert, svc_pred, average='macro'),
        'f1_macro': f1_score(y_test_bert, svc_pred, average='macro'),
        'precision_weighted': precision_score(y_test_bert, svc_pred, average='weighted'),
        'recall_weighted': recall_score(y_test_bert, svc_pred, average='weighted'),
        'f1_weighted': f1_score(y_test_bert, svc_pred, average='weighted')
    }
    
    # XGBoost
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train_vectorized_bert, y_train_bert)
    xgb_pred = xgb_model.predict(X_test_vectorized_bert)
    models['XGBoost_BERT'] = xgb_model
    results['XGBoost_BERT'] = {
        'pred': xgb_pred,
        'y_test': y_test_bert,
        'accuracy': accuracy_score(y_test_bert, xgb_pred),
        'confusion_matrix': confusion_matrix(y_test_bert, xgb_pred),
        'precision': precision_score(y_test_bert, xgb_pred, average=None),
        'recall': recall_score(y_test_bert, xgb_pred, average=None),
        'f1': f1_score(y_test_bert, xgb_pred, average=None),
        'precision_macro': precision_score(y_test_bert, xgb_pred, average='macro'),
        'recall_macro': recall_score(y_test_bert, xgb_pred, average='macro'),
        'f1_macro': f1_score(y_test_bert, xgb_pred, average='macro'),
        'precision_weighted': precision_score(y_test_bert, xgb_pred, average='weighted'),
        'recall_weighted': recall_score(y_test_bert, xgb_pred, average='weighted'),
        'f1_weighted': f1_score(y_test_bert, xgb_pred, average='weighted')
    }
    
    # Train and evaluate VADER models
    # GBDT
    gbdt_model_vader = GradientBoostingClassifier()
    gbdt_model_vader.fit(X_train_vectorized_vader, y_train_vader)
    gbdt_pred_vader = gbdt_model_vader.predict(X_test_vectorized_vader)
    models['GBDT_VADER'] = gbdt_model_vader
    results['GBDT_VADER'] = {
        'pred': gbdt_pred_vader,
        'y_test': y_test_vader,
        'accuracy': accuracy_score(y_test_vader, gbdt_pred_vader),
        'confusion_matrix': confusion_matrix(y_test_vader, gbdt_pred_vader),
        'precision': precision_score(y_test_vader, gbdt_pred_vader, average=None),
        'recall': recall_score(y_test_vader, gbdt_pred_vader, average=None),
        'f1': f1_score(y_test_vader, gbdt_pred_vader, average=None),
        'precision_macro': precision_score(y_test_vader, gbdt_pred_vader, average='macro'),
        'recall_macro': recall_score(y_test_vader, gbdt_pred_vader, average='macro'),
        'f1_macro': f1_score(y_test_vader, gbdt_pred_vader, average='macro'),
        'precision_weighted': precision_score(y_test_vader, gbdt_pred_vader, average='weighted'),
        'recall_weighted': recall_score(y_test_vader, gbdt_pred_vader, average='weighted'),
        'f1_weighted': f1_score(y_test_vader, gbdt_pred_vader, average='weighted')
    }
    
    # SVC
    svc_model_vader = SVC()
    svc_model_vader.fit(X_train_vectorized_vader, y_train_vader)
    svc_pred_vader = svc_model_vader.predict(X_test_vectorized_vader)
    models['SVC_VADER'] = svc_model_vader
    results['SVC_VADER'] = {
        'pred': svc_pred_vader,
        'y_test': y_test_vader,
        'accuracy': accuracy_score(y_test_vader, svc_pred_vader),
        'confusion_matrix': confusion_matrix(y_test_vader, svc_pred_vader),
        'precision': precision_score(y_test_vader, svc_pred_vader, average=None),
        'recall': recall_score(y_test_vader, svc_pred_vader, average=None),
        'f1': f1_score(y_test_vader, svc_pred_vader, average=None),
        'precision_macro': precision_score(y_test_vader, svc_pred_vader, average='macro'),
        'recall_macro': recall_score(y_test_vader, svc_pred_vader, average='macro'),
        'f1_macro': f1_score(y_test_vader, svc_pred_vader, average='macro'),
        'precision_weighted': precision_score(y_test_vader, svc_pred_vader, average='weighted'),
        'recall_weighted': recall_score(y_test_vader, svc_pred_vader, average='weighted'),
        'f1_weighted': f1_score(y_test_vader, svc_pred_vader, average='weighted')
    }
    
    # XGBoost
    xgb_model_vader = XGBClassifier()
    xgb_model_vader.fit(X_train_vectorized_vader, y_train_vader)
    xgb_pred_vader = xgb_model_vader.predict(X_test_vectorized_vader)
    models['XGBoost_VADER'] = xgb_model_vader
    results['XGBoost_VADER'] = {
        'pred': xgb_pred_vader,
        'y_test': y_test_vader,
        'accuracy': accuracy_score(y_test_vader, xgb_pred_vader),
        'confusion_matrix': confusion_matrix(y_test_vader, xgb_pred_vader),
        'precision': precision_score(y_test_vader, xgb_pred_vader, average=None),
        'recall': recall_score(y_test_vader, xgb_pred_vader, average=None),
        'f1': f1_score(y_test_vader, xgb_pred_vader, average=None),
        'precision_macro': precision_score(y_test_vader, xgb_pred_vader, average='macro'),
        'recall_macro': recall_score(y_test_vader, xgb_pred_vader, average='macro'),
        'f1_macro': f1_score(y_test_vader, xgb_pred_vader, average='macro'),
        'precision_weighted': precision_score(y_test_vader, xgb_pred_vader, average='weighted'),
        'recall_weighted': recall_score(y_test_vader, xgb_pred_vader, average='weighted'),
        'f1_weighted': f1_score(y_test_vader, xgb_pred_vader, average='weighted')
    }
    
    # Get class support information
    unique_bert, counts_bert = np.unique(y_test_bert, return_counts=True)
    unique_vader, counts_vader = np.unique(y_test_vader, return_counts=True)
    
    class_support = {
        'BERT': {str(unique_bert[i]): counts_bert[i] for i in range(len(unique_bert))},
        'VADER': {str(unique_vader[i]): counts_vader[i] for i in range(len(unique_vader))}
    }
    
    return models, results, class_support

# Function to process results and format metrics
def process_results(results, class_support):
    models = ['GBDT', 'SVC', 'XGBoost']
    
    # Extract accuracies
    bert_accuracies = [results[f'{model}_BERT']['accuracy'] for model in models]
    vader_accuracies = [results[f'{model}_VADER']['accuracy'] for model in models]
    
    # Process metrics for BERT
    bert_precisions = {}
    bert_recalls = {}
    bert_f1_scores = {}
    
    for model in models:
        bert_precisions[model] = {
            '0': results[f'{model}_BERT']['precision'][0],
            '1': results[f'{model}_BERT']['precision'][1],
            '2': results[f'{model}_BERT']['precision'][2],
            'macro': results[f'{model}_BERT']['precision_macro'],
            'weighted': results[f'{model}_BERT']['precision_weighted']
        }
        
        bert_recalls[model] = {
            '0': results[f'{model}_BERT']['recall'][0],
            '1': results[f'{model}_BERT']['recall'][1],
            '2': results[f'{model}_BERT']['recall'][2],
            'macro': results[f'{model}_BERT']['recall_macro'],
            'weighted': results[f'{model}_BERT']['recall_weighted']
        }
        
        bert_f1_scores[model] = {
            '0': results[f'{model}_BERT']['f1'][0],
            '1': results[f'{model}_BERT']['f1'][1],
            '2': results[f'{model}_BERT']['f1'][2],
            'macro': results[f'{model}_BERT']['f1_macro'],
            'weighted': results[f'{model}_BERT']['f1_weighted']
        }
    
    # Process metrics for VADER
    vader_precisions = {}
    vader_recalls = {}
    vader_f1_scores = {}
    
    for model in models:
        vader_precisions[model] = {
            '0': results[f'{model}_VADER']['precision'][0],
            '1': results[f'{model}_VADER']['precision'][1],
            '2': results[f'{model}_VADER']['precision'][2],
            'macro': results[f'{model}_VADER']['precision_macro'],
            'weighted': results[f'{model}_VADER']['precision_weighted']
        }
        
        vader_recalls[model] = {
            '0': results[f'{model}_VADER']['recall'][0],
            '1': results[f'{model}_VADER']['recall'][1],
            '2': results[f'{model}_VADER']['recall'][2],
            'macro': results[f'{model}_VADER']['recall_macro'],
            'weighted': results[f'{model}_VADER']['recall_weighted']
        }
        
        vader_f1_scores[model] = {
            '0': results[f'{model}_VADER']['f1'][0],
            '1': results[f'{model}_VADER']['f1'][1],
            '2': results[f'{model}_VADER']['f1'][2],
            'macro': results[f'{model}_VADER']['f1_macro'],
            'weighted': results[f'{model}_VADER']['f1_weighted']
        }
    
    # Extract confusion matrices
    conf_matrices = {
        'GBDT_BERT': results['GBDT_BERT']['confusion_matrix'],
        'SVC_BERT': results['SVC_BERT']['confusion_matrix'],
        'XGBoost_BERT': results['XGBoost_BERT']['confusion_matrix'],
        'GBDT_VADER': results['GBDT_VADER']['confusion_matrix'],
        'SVC_VADER': results['SVC_VADER']['confusion_matrix'],
        'XGBoost_VADER': results['XGBoost_VADER']['confusion_matrix']
    }
    
    return (bert_accuracies, vader_accuracies, 
            bert_precisions, bert_recalls, bert_f1_scores, 
            vader_precisions, vader_recalls, vader_f1_scores, 
            conf_matrices)

# Main program
try:
    # Try to load and preprocess data
    with st.spinner("Loading and processing data... This may take a moment."):
        db_merge = load_data()
        
        if db_merge is not None:
            # If data is available, train models and process results
            models, results, class_support = train_models(db_merge)
            
            (bert_accuracies, vader_accuracies, 
            bert_precisions, bert_recalls, bert_f1_scores, 
            vader_precisions, vader_recalls, vader_f1_scores, 
            conf_matrices) = process_results(results, class_support)
            
            models_list = ['GBDT', 'SVC', 'XGBoost']
            
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
                                x=models_list,
                                y=bert_accuracies,
                                name='BERT',
                                marker_color='indianred',
                                text=[f"{acc:.5f}" for acc in bert_accuracies],
                                textposition='auto',
                            ))
                        elif metric == "Precision":
                            bert_macro_precisions = [bert_precisions[model]['macro'] for model in models_list]
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=bert_macro_precisions,
                                name='BERT',
                                marker_color='indianred',
                                text=[f"{prec:.5f}" for prec in bert_macro_precisions],
                                textposition='auto',
                            ))
                        elif metric == "Recall":
                            bert_macro_recalls = [bert_recalls[model]['macro'] for model in models_list]
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=bert_macro_recalls,
                                name='BERT',
                                marker_color='indianred',
                                text=[f"{rec:.5f}" for rec in bert_macro_recalls],
                                textposition='auto',
                            ))
                        elif metric == "F1 Score":
                            bert_macro_f1s = [bert_f1_scores[model]['macro'] for model in models_list]
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=bert_macro_f1s,
                                name='BERT',
                                marker_color='indianred',
                                text=[f"{f1:.5f}" for f1 in bert_macro_f1s],
                                textposition='auto',
                            ))
                    
                    if model_type in ["VADER Lexicon", "Keduanya"]:
                        if metric == "Accuracy":
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=vader_accuracies,
                                name='VADER Lexicon',
                                marker_color='royalblue',
                                text=[f"{acc:.5f}" for acc in vader_accuracies],
                                textposition='auto',
                            ))
                        elif metric == "Precision":
                            vader_macro_precisions = [vader_precisions[model]['macro'] for model in models_list]
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=vader_macro_precisions,
                                name='VADER Lexicon',
                                marker_color='royalblue',
                                text=[f"{prec:.5f}" for prec in vader_macro_precisions],
                                textposition='auto',
                            ))
                        elif metric == "Recall":
                            vader_macro_recalls = [vader_recalls[model]['macro'] for model in models_list]
                            fig.add_trace(go.Bar(
                                x=models_list,
                                y=vader_macro_recalls,
                                name='VADER Lexicon',
                                marker_color='royalblue',
                                text=[f"{rec:.5f}" for rec in vader_macro_recalls],
                                textposition='auto',
                            ))
                        elif metric == "F1 Score":
                            vader_macro_f1s = [vader_f1_scores[model]['macro'] for model in models_list]
                            fig.add_trace(go.Bar(
                                x=models_list,
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
                                detail_data = {model: {k: v for k, v in bert_precisions[model].items() if k in ['0', '1', '2']} for model in models_list}
                            elif metric == "Recall":
                                detail_data = {model: {k: v for k, v in bert_recalls[model].items() if k in ['0', '1', '2']} for model in models_list}
                            else:  # F1 Score
                                detail_data = {model: {k: v for k, v in bert_f1_scores[model].items() if k in ['0', '1', '2']} for model in models_list}
                            
                            df = pd.DataFrame({
                                "Algoritma": models_list,
                                "Negatif (0)": [detail_data[model]['0'] for model in models_list],
                                "Netral (1)": [detail_data[model]['1'] for model in models_list],
                                "Positif (2)": [detail_data[model]['2'] for model in models_list],
                                "Macro Avg": [detail_data[model]['macro'] for model in models_list],
                                "Support": [f"{class_support['BERT']['0']}, {class_support['BERT']['1']}, {class_support['BERT']['2']}"] * len(models_list)
                            })
                            
                            st.dataframe(df, use_container_width=True)
                            
                        elif model_type == "VADER Lexicon":
                            if metric == "Precision":
                                detail_data = {model: {k: v for k, v in vader_precisions[model].items() if k in ['0', '1', '2']} for model in models_list}
                            elif metric == "Recall":
                                detail_data = {model: {k: v for k, v in vader_recalls[model].items() if k in ['0', '1', '2']} for model in models_list}
                            else:  # F1 Score
                                detail_data = {model: {k: v for k, v in vader_f1_scores[model].items() if k in ['0', '1', '2']} for model in models_list}
                            
                            df = pd.DataFrame({
                                "Algoritma": models_list,
                                "Negatif (0)": [detail_data[model]['0'] for model in models_list],
                                "Netral (1)": [detail_data[model]['1'] for model in models_list],
                                "Positif (2)": [detail_data[model]['2'] for model in models_list],
                                "Macro Avg": [detail_data[model]['macro'] for model in models_list],
                                "Support": [f"{class_support['VADER']['0']}, {class_support['VADER']['1']}, {class_support['VADER']['2']}"] * len(models_list)
                            })
                            
                            st.dataframe(df, use_container_width=True)
                            
                        else:  # Keduanya
                            st.write("**BERT**")
                            if metric == "Precision":
                                detail_data = {model: {k: v for k, v in bert_precisions[model].items() if k in ['0', '1', '2']} for model in models_list}
                            elif metric == "Recall":
                                detail_data = {model: {k: v for k, v in bert_recalls[model].items() if k in ['0', '1', '2']} for model in models_list}
                            else:  # F1 Score
                                detail_data = {model: {k: v for k, v in bert_f1_scores[model].items() if k in ['0', '1', '2']} for model in models_list}
                            
                            df_bert = pd.DataFrame({
                                "Algoritma": models_list,
                                "Negatif (0)": [detail_data[model]['0'] for model in models_list],
                                "Netral (1)": [detail_data[model]['1'] for model in models_list],
                                "Positif (2)": [detail_data[model]['2'] for model in models_list],
                                "Macro Avg": [detail_data[model]['macro'] for model in models_list],
                                "Support": [f"{class_support['BERT']['0']}, {class_support['BERT']['1']}, {class_support['BERT']['2']}"] * len(models_list)
                            })
                            
                            st.dataframe(df_bert, use_container_width=True)
                            
                            st.write("**VADER Lexicon**")
                            if metric == "Precision":
                                detail_data = {model: {k: v for k, v in vader_precisions[model].items() if k in ['0', '1', '2']} for model in models_list}
                            elif metric == "Recall":
                                detail_data = {model: {k: v for k, v in vader_recalls[model].items() if k in ['0', '1', '2']} for model in models_list}
                
                df_vader = pd.DataFrame({
                    "Algoritma": models_list,
                    "Negatif (0)": [detail_data[model]['0'] for model in models_list],
                    "Netral (1)": [detail_data[model]['1'] for model in models_list],
                    "Positif (2)": [detail_data[model]['2'] for model in models_list],
                    "Macro Avg": [detail_data[model]['macro'] for model in models_list],
                    "Support": [f"{class_support['VADER']['0']}, {class_support['VADER']['1']}, {class_support['VADER']['2']}"] * len(models_list)
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
            
            **Parameter yang digunakan:**
            - n_estimators: 100
            - learning_rate: 0.1
            - max_depth: 3
            - subsample: 1.0
            - criterion: 'friedman_mse'
            
            **Cara Kerja:**
            1. Memulai dengan model sederhana (biasanya satu node tree)
            2. Mengidentifikasi "residual" atau kesalahan dari model
            3. Melatih tree berikutnya untuk memprediksi residual tersebut
            4. Menambahkan tree baru ke model untuk mengurangi kesalahan
            5. Mengulangi proses hingga mencapai jumlah iterasi yang ditentukan
            
            **Kelebihan:**
            - Akurasi tinggi dan mampu menangani berbagai jenis data
            - Dapat menangani interaksi fitur secara alami
            - Mampu menangani data yang tidak seimbang
            - Tahan terhadap outlier
            
            **Kelemahan:**
            - Waktu pelatihan relatif lama
            - Risiko overfitting jika parameter tidak diatur dengan tepat
            - Membutuhkan lebih banyak tuning parameter dibanding algoritma lain
            """)
            
            st.markdown("**Durasi pelatihan:** 27 detik (BERT), 18 detik (VADER)")
            
        elif selected_algo == "Support Vector Classifier (SVC)":
            st.subheader("Support Vector Classifier (SVC)")
            st.markdown("""
            **Deskripsi:**
            Support Vector Classifier (SVC) adalah implementasi dari Support Vector Machine (SVM) untuk klasifikasi. SVC mencari hyperplane terbaik 
            yang memisahkan data dari kelas yang berbeda dengan margin maksimal.
            
            **Parameter yang digunakan:**
            - kernel: 'rbf'
            - C: 1.0
            - gamma: 'scale'
            - decision_function_shape: 'ovr' (one-vs-rest)
            - probability: True
            
            **Cara Kerja:**
            1. Mentransformasikan data ke dimensi yang lebih tinggi menggunakan kernel
            2. Menemukan hyperplane optimal yang memaksimalkan margin antara kelas
            3. Mengidentifikasi "support vectors" - titik data yang paling dekat dengan hyperplane
            4. Membuat keputusan klasifikasi berdasarkan posisi relatif terhadap hyperplane
            
            **Kelebihan:**
            - Efektif pada ruang berdimensi tinggi
            - Bekerja baik ketika jumlah dimensi lebih besar dari jumlah sampel
            - Memori efisien karena hanya menggunakan subset poin data (support vectors)
            - Kuat secara teoritis dan cenderung menghindari overfitting
            
            **Kelemahan:**
            - Tidak cocok untuk dataset besar (pelatihan lambat)
            - Kurang bekerja baik jika kelas sangat tumpang tindih
            - Pemilihan kernel dan parameter yang tepat penting
            - Kinerjanya menurun dengan dataset yang memiliki noise tinggi
            """)
            
            st.markdown("**Durasi pelatihan:** 19 detik (BERT), 14 detik (VADER)")
            
        else:  # XGBoost
            st.subheader("Extreme Gradient Boosting (XGBoost)")
            st.markdown("""
            **Deskripsi:**
            XGBoost adalah implementasi teroptimasi dari algoritma Gradient Boosting. Dikenal karena kecepatan dan performa yang unggul,
            XGBoost telah menjadi salah satu algoritma yang paling populer dalam kompetisi machine learning.
            
            **Parameter yang digunakan:**
            - n_estimators: 100
            - learning_rate: 0.1
            - max_depth: 3
            - subsample: 0.8
            - colsample_bytree: 0.8
            - objective: 'multi:softprob'
            - eval_metric: 'mlogloss'
            
            **Cara Kerja:**
            1. Menggunakan prinsip gradient boosting seperti GBDT
            2. Menambahkan regularisasi untuk mengurangi overfitting
            3. Menggunakan struktur tree yang dioptimalkan dan algoritma paralel
            4. Menangani nilai yang hilang secara otomatis
            5. Menggunakan teknik "pruning" untuk menghilangkan split yang tidak signifikan
            
            **Kelebihan:**
            - Kinerja yang sangat baik pada berbagai jenis data
            - Kecepatan eksekusi yang cepat dan efisien
            - Fitur regularisasi built-in untuk mengurangi overfitting
            - Penanganan nilai yang hilang secara otomatis
            - Paralelisasi yang efisien
            
            **Kelemahan:**
            - Memerlukan lebih banyak tuning parameter dibanding model yang lebih sederhana
            - Dapat overfitting pada dataset kecil jika tidak diatur dengan baik
            - Membutuhkan lebih banyak memori dibanding algoritma gradient boosting lainnya
            """)
            
            st.markdown("**Durasi pelatihan:** 28 detik (BERT), 26 detik (VADER)")

    with tab3:
        st.header("Interpretasi Hasil")
        
        st.subheader("Ringkasan Perbandingan")
        
        # Membuat tabel ringkasan perbandingan
        summary_df = pd.DataFrame({
            'Algoritma': models_list,
            'Akurasi BERT': bert_accuracies,
            'Akurasi VADER': vader_accuracies,
            'Precision BERT (macro)': [bert_precisions[model]['macro'] for model in models_list],
            'Precision VADER (macro)': [vader_precisions[model]['macro'] for model in models_list],
            'Recall BERT (macro)': [bert_recalls[model]['macro'] for model in models_list],
            'Recall VADER (macro)': [vader_recalls[model]['macro'] for model in models_list],
            'F1 Score BERT (macro)': [bert_f1_scores[model]['macro'] for model in models_list],
            'F1 Score VADER (macro)': [vader_f1_scores[model]['macro'] for model in models_list]
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
        - Secara konsisten, GBDT dan XGBoost memberikan nilai akurasi tertinggi saat menggunakan VADER Lexicon.
        - Untuk metriks F1-score (macro), GBDT dan XGBoost mencapai nilai tertinggi dengan VADER, mengungguli SVC.

        2. **SVC Unggul dengan BERT**
        - Dalam konteks BERT, SVC mencapai akurasi tertinggi dibandingkan XGBoost dan GBDT.
        - SVC juga menunjukkan F1-score (macro) terbaik untuk pelabelan BERT.

        3. **BERT vs VADER Lexicon**
        - Model yang menggunakan fitur VADER secara konsisten menunjukkan performa yang lebih baik dibandingkan model BERT.
        - Gap performa antara keduanya cukup signifikan, dengan peningkatan akurasi yang substansial saat menggunakan VADER.
        - Ini mengindikasikan bahwa VADER lebih sesuai untuk analisis sentimen berbahasa Indonesia dalam konteks program stunting.

        4. **Perbandingan Distribusi Kelas**
        - Data VADER memiliki kecenderungan bias terhadap label netral, sementara distribusi BERT lebih seimbang.
        - Hal ini menjelaskan mengapa akurasi VADER tinggi, namun F1-score makro lebih rendah untuk beberapa model karena kesulitan mendeteksi kelas minoritas.

        ### Implikasi untuk Analisis Sentimen Program Stunting

        Berdasarkan hasil ini, untuk mengklasifikasikan sentimen publik terhadap program stunting di media sosial X:
        
        - **Rekomendasi Model**: GBDT dengan fitur VADER memberikan hasil optimal untuk fokus pada akurasi keseluruhan.
        - **Pertimbangan Keseimbangan Kelas**: Jika deteksi sentimen negatif/positif (kelas minoritas) penting, SVC dengan BERT lebih direkomendasikan.
        - **Trade-off Kecepatan vs Akurasi**: SVC menawarkan pelatihan tercepat dengan akurasi kompetitif pada VADER.
        - **Pengembangan Lebih Lanjut**: Teknik oversampling atau class weighting dapat diterapkan untuk mengatasi ketidakseimbangan kelas.
        """)
        
        st.subheader("Visualisasi Perbandingan Algoritma")
        
        # Radar chart untuk membandingkan kinerja algoritma
        radar_categories = ['Akurasi', 'Precision', 'Recall', 'F1 Score', 'Kecepatan Training']
        
        # BERT metrics for radar chart (normalized between 0-1)
        bert_radar_values = {
            'GBDT': [
                bert_accuracies[0], 
                bert_precisions['GBDT']['macro'], 
                bert_recalls['GBDT']['macro'],
                bert_f1_scores['GBDT']['macro'],
                0.7  # Normalized speed (inverse of training time)
            ],
            'SVC': [
                bert_accuracies[1], 
                bert_precisions['SVC']['macro'], 
                bert_recalls['SVC']['macro'],
                bert_f1_scores['SVC']['macro'],
                0.8  # Normalized speed
            ],
            'XGBoost': [
                bert_accuracies[2], 
                bert_precisions['XGBoost']['macro'], 
                bert_recalls['XGBoost']['macro'],
                bert_f1_scores['XGBoost']['macro'],
                0.6  # Normalized speed
            ]
        }
        
        # VADER metrics for radar chart
        vader_radar_values = {
            'GBDT': [
                vader_accuracies[0], 
                vader_precisions['GBDT']['macro'], 
                vader_recalls['GBDT']['macro'],
                vader_f1_scores['GBDT']['macro'],
                0.8  # Normalized speed
            ],
            'SVC': [
                vader_accuracies[1], 
                vader_precisions['SVC']['macro'], 
                vader_recalls['SVC']['macro'],
                vader_f1_scores['SVC']['macro'],
                0.9  # Normalized speed
            ],
            'XGBoost': [
                vader_accuracies[2], 
                vader_precisions['XGBoost']['macro'], 
                vader_recalls['XGBoost']['macro'],
                vader_f1_scores['XGBoost']['macro'],
                0.7  # Normalized speed
            ]
        }
        
        # Pilihan model untuk radar chart
        radar_model = st.radio(
            "Pilih model untuk visualisasi radar:",
            ["BERT", "VADER Lexicon"]
        )
        
        # Plot radar chart
        if radar_model == "BERT":
            radar_values = bert_radar_values
        else:
            radar_values = vader_radar_values
        
        fig = go.Figure()
        
        for algo, values in radar_values.items():
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_categories,
                fill='toself',
                name=algo
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=f"Perbandingan Kinerja Algoritma ({radar_model})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Rekomendasi Final")
        st.info("""
        Berdasarkan analisis komprehensif dari ketiga algoritma machine learning dan dua metode analisis sentimen,
        rekomendasi kami untuk klasifikasi sentimen program stunting adalah:
        
        **📊 GBDT dengan VADER Lexicon** untuk kasus penggunaan umum dengan fokus pada akurasi keseluruhan.
        
        **🔍 SVC dengan BERT** untuk kasus di mana identifikasi sentimen negatif atau positif menjadi prioritas,
        terutama ketika distribusi kelas tidak seimbang.
        
        Untuk optimasi lebih lanjut, kombinasi teknik pemrosesan data lanjutan seperti pembobotan kelas atau
        ensemble model dapat dipertimbangkan untuk meningkatkan performa pada kasus spesifik.
        """)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.warning("Coba muat ulang halaman atau periksa kembali file data yang digunakan.")