# 💻 Source Code Directory

This directory contains all the Python source code for the Football Player Profiling project.

---

## 📂 Subdirectories and Contents

### 1. `data_collection/` - **Phase 1: Data Collection**
**Purpose**: Fetching and validating raw match data from StatsBomb.
- `collect_wc2022_events.py`: Collects events and match metadata for FIFA World Cup 2022.
- `validate_player_sample.py`: Verifies data quality and player sample sizes.

### 2. `feature_engineering/` - **Phase 2: KPI Calculation**
**Purpose**: Transforming raw event data into position-specific Key Performance Indicators (KPIs).
- `calculate_kpis.py`: Main script for calculating all positional metrics.
- `clean_kpi_data.py`: Prepares calculated KPIs for machine learning (handling outliers, normalization).

### 3. `clustering/` - **Phase 3.1: Player Profiling**
**Purpose**: Grouping players into tactical archetypes using K-Means clustering.
- `optimal_k_selection.py`: Determines the best number of clusters (K) using multi-metric validation (Silhouette, Elbow, etc.).
- `kmeans_clustering.py`: Implementation of K-Means with robustness testing (ARI).
- `cluster_profiling.py`: Generates statistical profiles and tactical names for each cluster.
- `cluster_visualization.py`: Creates PCA plots, radar charts, and heatmaps for analysis.

### 4. `feature_importance/` - **Phase 3.2: Model Validation**
**Purpose**: Validating clustering results and identifying key performance drivers.
- `rf_feature_importance.py`: Uses Random Forest to identify which KPIs define each cluster profile.

### 5. `app/` - **Phase 4: Dashboard**
**Purpose**: Interactive Streamlit web application for visualizing results.
- `Dashboard.py`: Main entry point for the Streamlit application.
- `cache_manager.py`: Handles data caching for optimized performance.
- `data_loader.py`: Specialized module for loading processed data into the UI.
- `pages/`: Additional dashboard views and analysis pages.

### 6. `utils/` - **Common Utilities**
**Purpose**: Shared helper functions used across all project phases.
- `kpi_helpers.py`: Common mathematical and data processing functions for KPIs.
- `position_mapping.py`: Logic for mapping raw StatsBomb positions to the 6 core analysis roles.

---

## 🎯 Pipeline Workflow

The project follows a sequential development pipeline:

```
1. data_collection/      → Raw Parquet/CSV files
       ↓
2. feature_engineering/  → Calculated KPI tables
       ↓
3. clustering/           → Player archetypes and models
       ↓
4. feature_importance/   → Model validation reports
       ↓
5. app/                  → Interactive Visualization
```

---

