import pandas as pd
import numpy as np
import re
import json
import time
import pickle
import h5py # NEW: For Snapchat
import os
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from google import genai
from umap import UMAP
from sentence_transformers import SentenceTransformer

class TopicAnalyst:
    def __init__(self, df, api_key, config, nr_topics=13, clustering_method='visual'):
        """
        config: Dictionary containing file paths and column mappings.
                Example: {
                    'col_id': 'Id',
                    'col_title': 'Title',
                    'col_desc': 'Description',
                    'col_path': 'video_path',
                    'visual_path': '...',
                    'desc_path': '...',
                    'dataset_type': 'snapchat' # or 'youtube'
                }
        """
        self.df = df.copy()
        self.cfg = config 
        self.nr_topics = nr_topics
        self.clustering_method = clustering_method
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"
        
        # Internal standard names
        self.COL_ID = self.cfg.get('col_id', 'id')
        self.COL_TITLE = self.cfg.get('col_title', 'title')
        self.COL_PATH = self.cfg.get('col_path', 'video_path')
        
        self.topic_model = None
        self.topics = None
        self.topic_info = None
        self.embeddings_used = None

    def _load_visuals(self):
        """Smart loader for H5 (Snap) or PKL (YouTube)"""
        fpath = self.cfg.get('path_visual')
        
        # If visual clustering is requested, we MUST have this file
        if self.clustering_method == 'visual':
            if not fpath or not os.path.exists(fpath):
                raise FileNotFoundError(f"Visual features file not found at: {fpath}")

        if not fpath or not os.path.exists(fpath):
            return

        print(f"Loading visuals from {fpath}...")
        visual_features_dict = {}

        try:
            if fpath.endswith('.h5'):
                with h5py.File(fpath, 'r') as f:
                    for video_id in f.keys():
                        visual_features_dict[str(video_id)] = f[video_id][:]
            else: # Assume Pickle
                with open(fpath, 'rb') as f:
                    loaded_data = pickle.load(f)
                    # Handle case where pickle is a list of dicts vs single dict
                    if isinstance(loaded_data, dict):
                        # Ensure keys are strings
                        visual_features_dict = {str(k): v for k, v in loaded_data.items()}
                    else:
                        print("Warning: Pickle format unknown (not a dict).")

            # Ensure DF IDs are strings to match Dict keys
            self.df[self.COL_ID] = self.df[self.COL_ID].astype(str).str.strip()
            
            # Map features
            self.df['visual'] = self.df[self.COL_ID].map(visual_features_dict)
            
            # Check success
            matched_count = self.df['visual'].notna().sum()
            print(f"Successfully mapped visual features for {matched_count} / {len(self.df)} videos.")
            
            # Raise error if mapping completely failed for Visual Clustering
            if self.clustering_method == 'visual' and matched_count == 0:
                raise ValueError("Visual mapping failed! 0 videos matched. Check if Video IDs in CSV match keys in Pickle/H5.")
                
            # Drop missing only if strictly using visual clustering
            if self.clustering_method == 'visual':
                self.df = self.df.dropna(subset=['visual'])
                
        except Exception as e:
            # Re-raise error if we need visuals, otherwise just warn
            if self.clustering_method == 'visual':
                raise e
            print(f"Warning skipping visuals: {e}")

    def _load_descriptions(self):
        """Parses JSONL based on dataset type"""
        fpath = self.cfg.get('path_desc')
        dtype = self.cfg.get('dataset_type', 'generic')
        
        if not fpath or not os.path.exists(fpath):
            return

        print(f"Loading descriptions ({dtype}) from {fpath}...")
        desc_map = {}
        
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        
                        # --- CUSTOM PARSING LOGIC ---
                        if dtype == 'snapchat':
                            
                            for vid_id, text in item.items():
                                if text != 'error':
                                    desc_map[str(vid_id)] = text
                                    
                        elif dtype == 'youtube':
                           
                            vid_id = item.get('video_name', '').split('.')[0]
                            text = item.get('Description', '')
                            if text and text != 'error':
                                desc_map[str(vid_id)] = text
                        # ----------------------------
                        
                    except: continue
            
            self.df['external_desc'] = self.df[self.COL_ID].astype(str).map(desc_map)
            print(f"Mapped {len(desc_map)} descriptions.")
            
        except Exception as e:
            print(f"Error loading description file: {e}")

    def preprocess(self):
        # 1. Load External Data
        self._load_visuals()
        self._load_descriptions()

        # 2. Determine Text Source
        # We look at self.cfg['text_source'] which comes from App selection
        target_mode = self.cfg.get('text_source', 'Title Only')
        
        # Get standardized Series for Title and Desc
        s_title = self.df[self.COL_TITLE].astype(str).fillna('')
        
        # Use external JSONL desc if available, else look for CSV column
        if 'external_desc' in self.df.columns:
            s_desc = self.df['external_desc'].fillna('')
        else:
            col_desc = self.cfg.get('col_desc', 'Description')
            if col_desc in self.df.columns:
                s_desc = self.df[col_desc].astype(str).fillna('')
            else:
                s_desc = pd.Series([""] * len(self.df))

        # 3. Construct Final Text
        if target_mode == 'Description':
            # Fallback to title if desc is empty
            self.df['final_text'] = s_desc
            self.df.loc[self.df['final_text'] == '', 'final_text'] = s_title
        elif target_mode == 'Title + Description':
            self.df['final_text'] = s_title + " : " + s_desc
        else: # Title Only
            self.df['final_text'] = s_title

        # 4. Clean
        def clean_text(text):
            if not isinstance(text, str): return ""
            text = re.sub(r'[^a-zA-Z0-9\s\.,;!?\'"-]', '', text) 
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        self.df['standardized_text'] = self.df['final_text'].apply(clean_text)
        self.df = self.df[self.df['standardized_text'].str.len() > 0].copy()
        
        return self.df

    def fit_model(self, progress_callback=None):
        docs = self.df['standardized_text'].tolist()
        
        # A. Embeddings
        if self.clustering_method == 'visual':
            if 'visual' not in self.df.columns:
                raise ValueError("Visual column missing.")
            embeddings = np.vstack(self.df['visual'].values)
        else:
            if progress_callback: progress_callback("Generating Text Embeddings...", 10)
            model = SentenceTransformer("all-MiniLM-L6-v2") 
            embeddings = model.encode(docs, show_progress_bar=False)

        self.embeddings_used = embeddings

        # B. BERTopic
        if progress_callback: progress_callback("Initializing Model...", 30)
        
        # Generic Stopwords
        stops = list(ENGLISH_STOP_WORDS) +[
            'shorts', 'short', 'video', 'vlog', 'tv', 'channel', 'subscribe', 'like', 'fyp',
            'shoz', 'beideo', 'yeongsang', 'beullogueu', 'chaeneol', 'gudog', 'joayo',
            'none', 'nan', 'make', 'get', 'do', 'go', 'know', 'look', 'see', 'watch', # Generic verbs
            "video", "clip", "footage", "camera", "lens", "screen",
            "person", "man", "woman", "people", "hand", "hands", "holding",
            "begins", "ends", "starts", "shows", "seen", "view",
            "table", "background", "foreground", "room", "revealing", "proceeds",
            "kg", "g", "lb", "lbs", "cm", "m", "mm", "km", "inch", "inches",
            "day", "days", "week", "weeks", "month", "months", "year", "years",
            "time", "times", "minute", "minutes", "second", "seconds",
            "new", "old", "first", "last", "next", "best", "top", "big", "small",
            'kcal', 'calorie', 'calories'
        ]
        vectorizer = CountVectorizer(stop_words=stops, min_df=5, ngram_range=(1, 2))
        mmr = MaximalMarginalRelevance(diversity=0.6)

        self.topic_model = BERTopic(
            embedding_model="all-mpnet-base-v2",
            vectorizer_model=vectorizer,
            representation_model=mmr,
            nr_topics=self.nr_topics,
            verbose=True
        )

        if progress_callback: progress_callback(f"Clustering ({self.clustering_method})...", 50)
        self.topics, _ = self.topic_model.fit_transform(docs, embeddings=embeddings)
        self.df['topic_id'] = self.topics
        self.topic_info = self.topic_model.get_topic_info()
        
        return self.topic_info

    def generate_labels(self, progress_callback=None):
        # ... (Same Gemini logic as before, just uses self.df['standardized_text']) ...
        # Copy previous generate_labels code here
        # Included briefly for completeness:
        unique_topics = sorted(list(set(self.topics)))
        label_map, desc_map = {}, {}
        total = len(unique_topics)
        
        for i, tid in enumerate(unique_topics):
            if progress_callback: progress_callback(f"Labeling {tid}...", 60 + int((i/total)*30))
            # if tid == -1:
            #     label_map[tid] = "Outliers"
            #     desc_map[tid] = "Mixed content"
            #     continue
            
            rep_docs = self.topic_model.get_representative_docs(tid)
            rep_docs = [d[:300] for d in rep_docs]
            keywords = [k[0] for k in self.topic_model.get_topic(tid)] if self.topic_model.get_topic(tid) else []
            

            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    prompt = f"""
                I have a cluster of videos grouped by VISUAL similarity. 
                Here are the titles of the most representative videos in this group: {json.dumps(rep_docs)}
                Here are the top keywords extracted from them: {keywords}

                The titles might be noisy (hashtags, bad grammar). 
                Identify the common visual theme (e.g., "Cooking", "Dance Cover", "Tech Review").

                IMPORTANT: Ensure the label is in standard English. If the inputs contain Korean or Romanized Korean words, translate the concept (e.g., use "Daily Vlog" instead of "Beullogueu").

                Return ONLY a JSON object with this format:
                {{
                    "topic_label": "Concise English Label (2-3 words)",
                    "description": "Short description of the video content (1 sentence)"
                }}
                """
                    
                    # Generate
                    res = self.client.models.generate_content(
                        model=self.model_name, 
                        contents=prompt,
                        # Relax safety slightly to prevent false blocks on social media content
                        config=types.GenerateContentConfig(
                            response_mime_type='application/json',
                            candidate_count=1
                        )
                    )
                    
                    data = self._clean_and_parse_json(res.text)
                    
                    if data:
                        label_map[tid] = data.get('topic_label', f"Topic {tid}")
                        description_map[tid] = data.get('description', '')
                        success = True
                        break # Exit retry loop
                    else:
                        raise ValueError("JSON parse failed")

                except Exception as e:
                    print(f"Attempt {attempt+1} failed for Topic {tid}: {e}")
                    time.sleep(2 * (attempt + 1)) # Exponential backoff: 2s, 4s, 6s

            if not success:
                # If all retries fail, give a descriptive error (visible in DataFrame)
                label_map[tid] = f"Topic {tid} (Failed)"
                desc_map[tid] = "Error: API could not label this cluster."

            

        self.df['topic_name'] = self.df['topic_id'].map(label_map)
        self.df['topic_description'] = self.df['topic_id'].map(desc_map)
        return self.df

    def get_visualization_data(self):
        umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
        reduced = umap_model.fit_transform(self.embeddings_used)
        viz_df = self.df.copy()
        viz_df['x'] = reduced[:, 0]
        viz_df['y'] = reduced[:, 1]
        return viz_df
