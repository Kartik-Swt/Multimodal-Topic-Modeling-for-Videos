import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Path Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from src.topic_model import TopicAnalyst
from src.utils import create_gif_preview

st.set_page_config(page_title="Multimodal Video Topics", layout="wide")

# --- DATASET REGISTRY ---
DATASET_REGISTRY = {
    "YouTube": {
        "path_csv": os.path.join(PROJECT_ROOT, "data/youtube/YouTubeShortsData.csv"),
        "path_visual": os.path.join(PROJECT_ROOT, "data/youtube/visual_features_resnet50.pkl"),
        "path_desc": os.path.join(PROJECT_ROOT, "data/youtube/descriptions.jsonl"),
        "dataset_type": "youtube",
        "col_id": "videoId",
        "col_title": "videoTitleEn",
        "col_desc": "videoDescEn",
        "col_path": "videoPath"
    },
    "Snapchat": {
        "path_csv": os.path.join(PROJECT_ROOT, "data/snapchat/SnapChatDataset.csv"),
        "path_visual": os.path.join(PROJECT_ROOT, "data/snapchat/resnet_features.h5"),
        "path_desc": os.path.join(PROJECT_ROOT, "data/snapchat/Snap_mPLUG_VideoDetails.jsonl"),
        "dataset_type": "snapchat",
        "col_id": "Id",
        "col_title": "Title",
        "col_desc": "Description",
        "col_path": "video_path"
    }
}

# --- Sidebar ---
st.sidebar.title("Config")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.sidebar.divider()

# 1. Dataset Selector
selected_dataset_name = st.sidebar.selectbox("Select Dataset", list(DATASET_REGISTRY.keys()))
dataset_config = DATASET_REGISTRY[selected_dataset_name].copy()

st.sidebar.info(f"Loaded config for: **{selected_dataset_name}**")

# 2. Text Source
text_source = st.sidebar.selectbox(
    "Text for Labeling",
    ["Title Only", "Description", "Title + Description"],
    index=1
)
dataset_config['text_source'] = text_source

# 3. Clustering Method
clustering_source = st.sidebar.radio("Clustering Method", ["Text Embeddings", "Visual Embeddings"])
method_code = 'text' if "Text" in clustering_source else 'visual'

nr_topics = st.sidebar.slider("Topics", 5, 30, 10)

# --- Main App ---
st.title("🎬 Multimodal Video Topic Modeler")

if st.button("Run Model"):
    if not api_key:
        st.error("Missing Gemini API Key")
    elif not os.path.exists(dataset_config['path_csv']):
        st.error(f"CSV not found: {dataset_config['path_csv']}")
    else:
        status = st.empty()
        progress = st.progress(0)
        
        def update_bar(text, val):
            status.text(text)
            progress.progress(val)

        try:
            # Load CSV
            df_raw = pd.read_csv(dataset_config['path_csv'])
            
            # Initialize Analyst
            analyst = TopicAnalyst(
                df_raw, 
                api_key, 
                config=dataset_config,
                nr_topics=nr_topics,
                clustering_method=method_code
            )
            
            analyst.preprocess()
            analyst.fit_model(progress_callback=update_bar)
            df_final = analyst.generate_labels(progress_callback=update_bar)
            
            st.session_state.df_result = df_final
            st.session_state.analyst = analyst
            st.session_state.dataset_cfg = dataset_config
            
            status.text("Done!")
            progress.progress(100)
            
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

# --- Results ---
if 'df_result' in st.session_state and st.session_state.df_result is not None:
    df = st.session_state.df_result
    cfg = st.session_state.dataset_cfg
    
    col_id = cfg['col_id']
    col_title = cfg['col_title']
    col_path = cfg['col_path']

    # NEW: Three Tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Topic Gallery", "📊 Analytics", "🗺️ Visual Map"])
    
    # --- Tab 1: Gallery ---
    with tab1:
        # topics = sorted([t for t in df['topic_id'].unique() if t != -1])
        topics = sorted([t for t in df['topic_id'].unique()])
        if not topics:
             st.warning("Only outliers found.")
        else:
            sel_topic = st.selectbox("Select Topic", topics, format_func=lambda x: f"{x}: {df[df['topic_id']==x]['topic_name'].iloc[0]}")
            
            subset = df[df['topic_id'] == sel_topic]
            if not subset.empty:
                st.info(f"**Description:** {subset['topic_description'].iloc[0]}")
                
                cols = st.columns(4)
                for i, (_, row) in enumerate(subset.head(4).iterrows()):
                    with cols[i%4]:
                        st.caption(f"**{str(row[col_title])[:40]}...**")
                        # GIF Preview
                        if col_path in row and os.path.exists(str(row[col_path])):
                            gif = create_gif_preview(row[col_path])
                            if gif: st.image(gif)
                        else:
                            st.warning("No Video")

    # --- Tab 2: Analytics ---
    with tab2:
        # 1. Bar Chart
        counts = df.groupby('topic_name').size().reset_index(name='Count')
        fig = px.bar(counts, x='topic_name', y='Count', title="Topic Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # 2. Filtered Dataframe
        st.subheader("Topic Data")
        
        # Select specific columns based on your request
        # We try to grab 'external_desc' first (from JSONL), otherwise fall back to CSV desc
        desc_col_to_show = 'external_desc' if 'external_desc' in df.columns else cfg['col_desc']
        
        # Build list of columns to display
        display_cols = [col_id, col_title, 'topic_name']
        
        # Add description if it exists in DF
        if desc_col_to_show in df.columns:
            display_cols.insert(2, desc_col_to_show)
            
        # Display
        st.dataframe(df[display_cols], use_container_width=True)

    # --- Tab 3: Visual Map ---
    with tab3:
        if st.button("Compute Visual Map"):
            with st.spinner("Projecting embeddings..."):
                # Get UMAP data from Analyst
                viz_df = st.session_state.analyst.get_visualization_data()
                
                fig = px.scatter(
                    viz_df, 
                    x='x', 
                    y='y', 
                    color='topic_name',
                    hover_data=[col_title, 'topic_name'], 
                    height=700,
                    title=f"Projection of {clustering_source}"
                )
                st.plotly_chart(fig, use_container_width=True)
