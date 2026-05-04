# 🎬 Multimodal Topic Modeling for Videos

An unsupervised topic modeling framework that fuses **visual features** (ResNet-50 backbone) with **textual data** (BERTopic + Sentence Transformers) to discover and label thematic clusters in large-scale short-form video datasets. Zero-shot topic labels are generated with **Google Gemini**, and results are explored through an interactive **Streamlit** dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Dual Clustering Modes** | Cluster by visual (ResNet-50) embeddings *or* text (Sentence Transformer) embeddings |
| **BERTopic Integration** | Leverages Maximal Marginal Relevance for diverse, meaningful keyword extraction |
| **Gemini Zero-Shot Labeling** | Automatically names each topic cluster with a concise English label and description |
| **Multi-Dataset Support** | Works with **YouTube Shorts** and **Snapchat** datasets out of the box |
| **Interactive Dashboard** | Three-tab Streamlit UI — Topic Gallery, Analytics, and UMAP Visual Map |
| **GIF Preview** | Renders short animated previews from video files directly in the UI |

---

## 🏗️ Architecture

```
Video Dataset (CSV + visual features + JSONL descriptions)
        │
        ▼
┌─────────────────────────────────────────┐
│             TopicAnalyst                │
│  ┌──────────┐    ┌────────────────────┐ │
│  │ Preprocess│    │  Fit BERTopic      │ │
│  │ text/desc │───▶│  (visual or text   │ │
│  │ cleaning  │    │   embeddings)      │ │
│  └──────────┘    └────────────────────┘ │
│                          │              │
│                  ┌───────▼────────┐     │
│                  │ Gemini Labeling│     │
│                  │ (zero-shot)    │     │
│                  └───────┬────────┘     │
└──────────────────────────┼─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Streamlit  │
                    │  Dashboard  │
                    │ ┌─────────┐ │
                    │ │ Gallery │ │
                    │ ├─────────┤ │
                    │ │Analytics│ │
                    │ ├─────────┤ │
                    │ │UMAP Map │ │
                    │ └─────────┘ │
                    └─────────────┘
```

### Key Components

- **`src/topic_model.py` — `TopicAnalyst` class**
  - `preprocess()` — Loads visual features (`.pkl` / `.h5`), JSONL descriptions, builds `final_text`, and cleans it.
  - `fit_model()` — Runs BERTopic with the chosen embedding type (visual ResNet-50 vectors **or** `all-MiniLM-L6-v2` sentence embeddings).
  - `generate_labels()` — Sends representative docs and keywords to Gemini-2.5-Flash to produce human-readable topic names and descriptions with retry logic.
  - `get_visualization_data()` — Projects embeddings to 2-D via UMAP for scatter-plot exploration.

- **`src/utils.py` — `create_gif_preview()`**
  - Extracts the first N seconds of a video file and converts them to a cached GIF using OpenCV + imageio.

- **`app.py` — Streamlit UI**
  - Sidebar controls: dataset selector, Gemini API key, text source, clustering method, and number of topics.
  - **Tab 1 – Topic Gallery**: Browse topics and preview up to 4 representative videos per topic.
  - **Tab 2 – Analytics**: Bar chart of topic distribution and a filterable DataFrame.
  - **Tab 3 – Visual Map**: On-demand UMAP scatter plot coloured by topic.

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit application entry point
├── readme.md               # This file
└── src/
    ├── topic_model.py      # TopicAnalyst class (core pipeline)
    └── utils.py            # GIF preview helper
```

Expected data layout (not committed to the repo):

```
data/
├── youtube/
│   ├── YouTubeShortsData.csv          # Video metadata
│   ├── visual_features_resnet50.pkl   # ResNet-50 feature vectors (dict: videoId → np.array)
│   └── descriptions.jsonl             # mPLUG/caption outputs per video
└── snapchat/
    ├── SnapChatDataset.csv            # Video metadata
    ├── resnet_features.h5             # ResNet-50 feature vectors (HDF5)
    └── Snap_mPLUG_VideoDetails.jsonl  # mPLUG/caption outputs per video
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- A [Google Gemini API key](https://aistudio.google.com/)

### 1. Clone the repository

```bash
git clone https://github.com/Kartik-Swt/Multimodal-Topic-Modeling-for-Videos.git
cd Multimodal-Topic-Modeling-for-Videos
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy plotly bertopic sentence-transformers \
            google-genai umap-learn scikit-learn h5py opencv-python imageio
```

> **Optional – GPU acceleration**: Install `torch` with CUDA before the above packages to speed up Sentence Transformer inference.

---

## ▶️ Usage

### Running the Dashboard

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (default: `http://localhost:8501`).

### Dashboard Walkthrough

1. **Enter your Gemini API Key** in the sidebar (stored in memory only, never saved).
2. **Select a Dataset** — `YouTube` or `Snapchat`.
3. **Choose a Text Source** for labeling:
   - *Title Only* — fastest, uses only the video title.
   - *Description* — uses AI-generated video descriptions from JSONL.
   - *Title + Description* — combines both.
4. **Select Clustering Method**:
   - *Text Embeddings* — clusters videos by semantic meaning of their text.
   - *Visual Embeddings* — clusters videos by visual appearance (requires pre-extracted ResNet features).
5. **Set the number of topics** using the slider (5–30).
6. Click **Run Model** and wait for the pipeline to complete.
7. Explore results across the three tabs.

### Using the Core Pipeline Programmatically

```python
import pandas as pd
from src.topic_model import TopicAnalyst

config = {
    "path_csv": "data/snapchat/SnapChatDataset.csv",
    "path_visual": "data/snapchat/resnet_features.h5",
    "path_desc": "data/snapchat/Snap_mPLUG_VideoDetails.jsonl",
    "dataset_type": "snapchat",
    "col_id": "Id",
    "col_title": "Title",
    "col_desc": "Description",
    "col_path": "video_path",
    "text_source": "Title + Description",
}

df = pd.read_csv(config["path_csv"])

analyst = TopicAnalyst(
    df,
    api_key="YOUR_GEMINI_API_KEY",
    config=config,
    nr_topics=12,
    clustering_method="visual",   # or "text"
)

analyst.preprocess()
analyst.fit_model()
df_result = analyst.generate_labels()

print(df_result[["Id", "Title", "topic_name", "topic_description"]].head())
```

---

## 🗃️ Dataset Format

### CSV

Must contain columns for video ID, title, description, and (optionally) local video file path. The exact column names are configured via `col_id`, `col_title`, `col_desc`, and `col_path` in the dataset config dictionary.

### Visual Features

Pre-extracted ResNet-50 feature vectors, one vector per video:

| Format | Structure |
|---|---|
| **Pickle (`.pkl`)** | `dict` mapping `video_id (str) → np.ndarray` |
| **HDF5 (`.h5`)** | Each top-level key is a `video_id`, value is the feature array |

### Descriptions (JSONL)

One JSON object per line describing video content (e.g., produced by a video captioning model like mPLUG-Owl):

- **Snapchat format**: `{"<video_id>": "<description text>", ...}` (multiple entries per line)
- **YouTube format**: `{"video_name": "<id>.mp4", "Description": "<text>"}` (one entry per line)

---

## ⚙️ Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nr_topics` | `int` | `13` | Target number of topic clusters |
| `clustering_method` | `str` | `"visual"` | `"visual"` or `"text"` |
| `text_source` | `str` | `"Title Only"` | `"Title Only"`, `"Description"`, or `"Title + Description"` |
| `dataset_type` | `str` | `"generic"` | `"youtube"` or `"snapchat"` (controls JSONL parsing) |

---

## 🔑 API Key

The Gemini API key is entered at runtime via the sidebar and is only used for the `generate_labels()` call. It is never persisted to disk.

Get a free key at [Google AI Studio](https://aistudio.google.com/).

---

## 📊 Results Overview

The pipeline produces a DataFrame where each video is annotated with:

| Column | Description |
|---|---|
| `topic_id` | Integer cluster ID assigned by BERTopic (`-1` = outlier) |
| `topic_name` | Short English label generated by Gemini (e.g., *"Cooking Tutorial"*) |
| `topic_description` | One-sentence description of the cluster's visual/textual theme |

---

## 🙏 Acknowledgements

- [BERTopic](https://maartengr.github.io/BERTopic/) by Maarten Grootendorst
- [Sentence Transformers](https://www.sbert.net/) by UKPLab
- [Google Gemini](https://deepmind.google/technologies/gemini/) for zero-shot labeling
- [UMAP](https://umap-learn.readthedocs.io/) for dimensionality reduction
- [Streamlit](https://streamlit.io/) for the interactive dashboard
