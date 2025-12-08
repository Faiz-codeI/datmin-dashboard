# Dashboard Streamlit untuk Klasifikasi Email Spam vs Ham

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
from bokeh.plotting import figure, from_networkx
from bokeh.models import (ColumnDataSource, HoverTool, LabelSet, 
                          Scatter, MultiLine, NodesAndLinkedEdges)
from bokeh.transform import cumsum
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Email Spam Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Minimalis
st.markdown("""
<style>
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & MODEL (CACHED)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('emails.zip', compression='zip')
        if "total_words" not in df.columns:
            cols_to_sum = [c for c in df.columns if c not in (df.columns[0], "Prediction")]
            df["total_words"] = df[cols_to_sum].sum(axis=1)
        return df
    except:
        return None

@st.cache_resource
def load_models():
    try:
        with open("xgb_model.pkl", "rb") as f: xgb = pickle.load(f)
        with open("nb_model.pkl", "rb") as f: nb = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f: vec = pickle.load(f)
        return xgb, nb, vec
    except:
        return None, None, None

df = load_data()
xgb_model, nb_model, vectorizer = load_models()

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("📧 Spam Dashboard")
    st.markdown("Visualisasi dataset klasifikasi Email Spam vs Ham.")
    
    if df is not None:
        total_email = len(df)
        total_spam = int((df["Prediction"] == 1).sum())
        total_ham = int((df["Prediction"] == 0).sum())
        
        st.write("---")
        st.metric("Total Email", f"{total_email}")
        st.metric("Total Spam", f"{total_spam}")
        st.metric("Total Ham", f"{total_ham}")
    
    st.write("---")
    st.caption("Created with Streamlit & Bokeh")

# ==========================================
# 4. MAIN CONTENT
# ==========================================

if df is None:
    st.error("File 'emails.zip' tidak terbaca. Pastikan file diupload dengan benar.")
else:
    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Email", len(df))
    with col2:
        st.metric("Jumlah Fitur", df.shape[1] - 2)
    with col3:
        spam_rate = (df["Prediction"] == 1).mean() * 100
        st.metric("Spam Rate", f"{spam_rate:.1f}%")
    with col4:
        st.metric("Avg Word Count", int(df["total_words"].mean()))

    st.markdown("---")

    # --- ROW 1: Donut & Histogram ---
    c1, c2 = st.columns(2)
    
    # CHART 1: DONUT
    with c1:
        st.subheader("Proporsi Kelas")
        
        # Data untuk Plotly Pie Chart
        labels = ['Ham', 'Spam']
        values = [total_ham, total_spam]
        colors = ['#4ade80', '#a78bfa']
    
        # Buat figure
        fig = go.Figure(data=[go.Pie(labels=labels, 
                                     values=values, 
                                     hole=.4, # Ini untuk membuatnya jadi donat
                                     marker_colors=colors,
                                     textinfo='percent+label',
                                     insidetextorientation='radial')])
    
        # Atur layout agar terlihat minimalis
        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)', # Transparan
            plot_bgcolor='rgba(0,0,0,0)'  # Transparan
        )
        
        # Tampilkan chart menggunakan Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # CHART 2: HISTOGRAM
    with c2:
        st.subheader("Distribusi Panjang Kata")
        words_ham = df[df['Prediction'] == 0]['total_words']
        words_spam = df[df['Prediction'] == 1]['total_words']
    
        # Buat figure Plotly
        fig2 = go.Figure()
    
        # Tambahkan trace untuk Ham dan Spam
        fig2.add_trace(go.Histogram(x=words_ham, name='Ham', marker_color='#4ade80'))
        fig2.add_trace(go.Histogram(x=words_spam, name='Spam', marker_color='#a78bfa'))
    
        # Atur layout agar histogram tumpang tindih (overlay) dan terlihat bagus
        fig2.update_layout(
            barmode='overlay',
            xaxis_title_text='Word Count',
            yaxis_title_text='Frekuensi',
            legend=dict(x=0.8, y=0.99),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        # Kurangi opacity agar bisa melihat tumpukan
        fig2.update_traces(opacity=0.6)
    
        st.plotly_chart(fig2, use_container_width=True)

    # --- ROW 2: Diverging Bar & Network ---
    c3, c4 = st.columns(2)    
    # CHART 3: DIVERGING BAR
    with c3:
        st.subheader("Kata Paling Pembeda")
        X = df.drop(columns=[df.columns[0], 'Prediction', 'total_words'], errors='ignore')
        mean_spam = X[df['Prediction'] == 1].mean()
        mean_ham  = X[df['Prediction'] == 0].mean()
        diff = mean_spam - mean_ham
        diverging = pd.concat([diff.sort_values().head(8), diff.sort_values().tail(8)])
        
        # Data untuk Plotly
        words = diverging.index.tolist()
        scores = diverging.values.tolist()
        colors = ["#4ade80" if x < 0 else "#a78bfa" for x in scores]
    
        # Buat figure
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=words,
            x=scores,
            orientation='h', # Membuat bar menjadi horizontal
            marker_color=colors
        ))
    
        # Tambahkan garis tengah
        fig3.add_shape(
            type="line",
            x0=0, y0=-0.5, x1=0, y1=len(words)-0.5,
            line=dict(color="Gray", width=1, dash="dash")
        )
    
        # Atur layout
        fig3.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Score Pembeda"
        )
        
        st.plotly_chart(fig3, use_container_width=True)

    # CHART 4: NETWORK GRAPH
    with c4:
        st.subheader("Jejaring Kata (Top 20)")
        top_20 = X.sum().sort_values(ascending=False).head(20).index
        subset_df = X[top_20]
        corr_matrix = subset_df.corr()
        
        G = nx.Graph()
        for w in top_20: G.add_node(w)
        for i in range(len(top_20)):
            for j in range(i+1, len(top_20)):
                if corr_matrix.iloc[i,j] > 0.3:
                    G.add_edge(top_20[i], top_20[j])
        
        pos = nx.spring_layout(G, k=0.3, seed=42)
    
        # Buat trace untuk garis (edges)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
    
        # Buat trace untuk titik (nodes)
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
    
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='#6366f1',
                size=15,
                line_width=2))
    
        # Buat figure dan gabungkan
        fig4 = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                        ))
        
        st.plotly_chart(fig4, use_container_width=True)

    # ==========================================
    # 5. PREDICTION SECTION
    # ==========================================
    st.header("🔮 Prediksi Email")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        user_input = st.text_area("Masukkan teks email di sini:", height=150, 
                                  placeholder="Congratulations! You won a lottery...")
        predict_btn = st.button("Analisis Email", type="primary")

    with col_result:
        if predict_btn and user_input:
            if xgb_model is None:
                st.error("Model tidak ditemukan atau rusak.")
            else:
                try:
                    txt_vec = vectorizer.transform([user_input]).toarray()
                    prob_xgb = xgb_model.predict_proba(txt_vec)[0]
                    prob_nb = nb_model.predict_proba(txt_vec)[0]
                    
                    st.subheader("XGBoost")
                    spam_prob_xgb = prob_xgb[1]
                    if spam_prob_xgb > 0.5:
                        st.error(f"SPAM DETECTED ({spam_prob_xgb*100:.1f}%)")
                    else:
                        st.success(f"HAM / AMAN ({spam_prob_xgb*100:.1f}%)")
                    st.progress(float(spam_prob_xgb))

                    st.subheader("Naive Bayes")
                    spam_prob_nb = prob_nb[1]
                    if spam_prob_nb > 0.5:
                        st.write(f"⚠️ Spam Prob: {spam_prob_nb*100:.1f}%")
                    else:
                        st.write(f"✅ Spam Prob: {spam_prob_nb*100:.1f}%")
                    st.progress(float(spam_prob_nb))
                except Exception as e:
                    st.error(f"Terjadi error data: {e}. \n\n**SOLUSI:** Generate ulang file .pkl (vectorizer & model) di laptop, lalu upload ulang ke GitHub.")
