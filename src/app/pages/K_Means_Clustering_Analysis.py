"""
Cluster Profiles - Modern Dashboard
------------------------------------
Interactive cluster exploration with PCA scatter, heatmaps, and player lists.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import config
from utils.ui import (
    display_breadcrumb,
    display_position_badge,
    inject_presentation_mode_css,
    display_contextual_help,
    display_loading_skeleton
)
from utils.charts import add_download_button
from data_loader import (
    load_clustered_data,
    load_pca_coordinates,
    load_tactical_names,
    load_z_scores,
    get_cluster_players
)

# ============================================================================
# MODERN CSS STYLES
# ============================================================================

st.markdown("""
<style>
/* Page Header */
.page-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-subtitle {
    color: #a8dadc;
    font-size: 0.95rem;
}

/* Metric Card */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f093fb, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    color: #888;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Cluster Card - Enhanced */
.cluster-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.cluster-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #f093fb, #f5576c);
}

.cluster-card:hover {
    border-color: rgba(240, 147, 251, 0.5);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(240, 147, 251, 0.15);
}

.cluster-card.selected {
    border-color: #f093fb;
    background: linear-gradient(145deg, rgba(240, 147, 251, 0.15) 0%, rgba(240, 147, 251, 0.05) 100%);
}

.cluster-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.8rem;
}

.cluster-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background: linear-gradient(135deg, #f093fb, #f5576c);
}

.cluster-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
}

.cluster-badge {
    display: inline-block;
    background: rgba(240, 147, 251, 0.2);
    color: #f093fb;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

.cluster-stats {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.5rem;
    margin-top: 0.8rem;
}

.cluster-stat {
    background: rgba(0,0,0,0.2);
    padding: 0.5rem;
    border-radius: 8px;
    text-align: center;
}

.cluster-stat-value {
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
}

.cluster-stat-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
}

/* Section Title */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(240, 147, 251, 0.3);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Heatmap Container */
.heatmap-container {
    background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.heatmap-legend {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1rem;
    padding: 0.8rem;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: #888;
}

.legend-color {
    width: 20px;
    height: 20px;
    border-radius: 4px;
}

/* Info Box */
.info-box {
    background: rgba(240, 147, 251, 0.1);
    border-left: 3px solid #f093fb;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    font-size: 0.9rem;
}

/* Player Table Container */
.player-table-container {
    background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}

.player-table-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

/* KPI Tag */
.kpi-tag {
    display: inline-block;
    background: rgba(240, 147, 251, 0.15);
    color: #f093fb;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 0.2rem;
    border: 1px solid rgba(240, 147, 251, 0.3);
}

/* Tab Style */
.cluster-tabs {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.cluster-tab {
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #888;
}

.cluster-tab:hover {
    background: rgba(240, 147, 251, 0.1);
    border-color: rgba(240, 147, 251, 0.3);
    color: #fff;
}

.cluster-tab.active {
    background: linear-gradient(90deg, #f093fb, #f5576c);
    border-color: transparent;
    color: #fff;
}

/* Variance Card */
.variance-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.variance-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f093fb;
}

.variance-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Description Box */
.description-box {
    background: rgba(0,0,0,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}

.description-title {
    font-size: 0.85rem;
    color: #f093fb;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.description-text {
    color: #ccc;
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'selected_position' not in st.session_state:
    st.session_state.selected_position = config.UI_SETTINGS['default_position']
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

inject_presentation_mode_css()

# Header
st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Cluster Profiles</div>
    <div class="page-subtitle">
        Visualize K-Means Clustering results with PCA scatter plots and z-score heatmaps
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# POSITION SELECTION
# ============================================================================

selected_position = st.session_state.selected_position
pos_info = config.POSITIONS[selected_position]

# Position Header
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.8rem;">
        <span style="font-size: 2rem;">{pos_info['icon']}</span>
        <div>
            <div style="font-size: 1.3rem; font-weight: 600;">{pos_info['name']}</div>
            <div style="color: #888; font-size: 0.85rem;">K-Means Clustering Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pos_info['n_players']}</div>
        <div class="metric-label">Players</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pos_info['n_clusters']}</div>
        <div class="metric-label">Clusters</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

clustered_df = load_clustered_data(selected_position)
pca_df = load_pca_coordinates(selected_position)
tactical_names = load_tactical_names(selected_position)
z_scores_df = load_z_scores(selected_position)

# ============================================================================
# PCA SCATTER PLOT
# ============================================================================

st.markdown('<div class="section-title">🔬 PCA Scatter Plot</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>💡 Chart Controls:</strong> Drag to pan • Scroll to zoom • Double-click to reset • Click legend to toggle clusters
</div>
""", unsafe_allow_html=True)

def create_pca_scatter(pca_df: pd.DataFrame, tactical_names: Dict, theme: str = 'dark') -> go.Figure:
    theme_config = config.THEMES[theme]
    cluster_colors = ['#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    
    pca_df = pca_df.copy()
    pca_df['cluster_name'] = pca_df['cluster_label'].apply(
        lambda x: tactical_names.get(x, {}).get('name', f'Cluster {x}')
    )

    fig = px.scatter(
        pca_df,
        x='pca_x',
        y='pca_y',
        color='cluster_name',
        hover_data=['player_name', 'cluster_label'],
        color_discrete_sequence=cluster_colors,
        labels={
            'pca_x': f'PC1 ({pca_df["explained_variance_ratio_1"].iloc[0]:.1%})',
            'pca_y': f'PC2 ({pca_df["explained_variance_ratio_2"].iloc[0]:.1%})',
            'cluster_name': 'Cluster'
        }
    )

    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='white')),
        hovertemplate="<b>%{customdata[0]}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
    )

    fig.update_layout(
        height=550,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=theme_config['text'])
        ),
        margin=dict(t=20, b=80, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=theme_config['text']),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )

    return fig

pca_fig = create_pca_scatter(pca_df, tactical_names, st.session_state.theme)
st.plotly_chart(pca_fig, use_container_width=True)
add_download_button(pca_fig, 'pca_scatter', 'PCA_Cluster_Plot')

# Explained Variance Metrics - Enhanced
exp_var_1 = pca_df['explained_variance_ratio_1'].iloc[0]
exp_var_2 = pca_df['explained_variance_ratio_2'].iloc[0]
total_var = exp_var_1 + exp_var_2

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="variance-card">
        <div class="variance-value">{exp_var_1:.1%}</div>
        <div class="variance-label">PC1 Variance</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="variance-card">
        <div class="variance-value">{exp_var_2:.1%}</div>
        <div class="variance-label">PC2 Variance</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="variance-card">
        <div class="variance-value">{total_var:.1%}</div>
        <div class="variance-label">Total Explained</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Z-SCORE HEATMAP - Enhanced
# ============================================================================

st.markdown("---")
st.markdown('<div class="section-title">🌡️ Z-Score Heatmap</div>', unsafe_allow_html=True)

st.markdown("""
<div class="heatmap-container">
    <div style="color: #ccc; margin-bottom: 1rem;">
        Standardized cluster performance comparison. Each cell shows how much a cluster 
        deviates from the average for that KPI.
    </div>
    <div class="heatmap-legend">
        <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(90deg, #d73027, #f46d43);"></div>
            <span>Above Average (+)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ffffbf;"></div>
            <span>Average (0)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: linear-gradient(90deg, #4575b4, #74add1);"></div>
            <span>Below Average (-)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

available_kpis = pos_info['kpis']
default_heatmap_kpis = available_kpis[:min(10, len(available_kpis))]

col1, col2 = st.columns([3, 1])

with col1:
    selected_heatmap_kpis = st.multiselect(
        "Select KPIs for Heatmap",
        options=available_kpis,
        default=default_heatmap_kpis,
        format_func=lambda x: config.KPI_READABLE_NAMES.get(x, x),
        label_visibility="collapsed"
    )

with col2:
    st.caption(f"📊 {len(selected_heatmap_kpis)} KPIs selected")

if len(selected_heatmap_kpis) >= 3:
    def create_z_score_heatmap(z_scores_df: pd.DataFrame, tactical_names: Dict, kpis: List[str], theme: str = 'dark') -> go.Figure:
        if 'z_score' not in z_scores_df.columns and 'cluster_id' in z_scores_df.columns:
            z_scores_df = z_scores_df.melt(id_vars=['cluster_id'], var_name='kpi', value_name='z_score')

        if 'kpi' not in z_scores_df.columns:
            z_scores_df = z_scores_df.reset_index().rename(columns={'index': 'kpi'})
            
        value_col = 'z_score' if 'z_score' in z_scores_df.columns else z_scores_df.select_dtypes(include=['float', 'int']).columns[0]
                    
        z_subset = z_scores_df[z_scores_df['kpi'].isin(kpis)].copy()
        z_matrix = z_subset.pivot(index='kpi', columns='cluster_id', values=value_col)
        kpi_variance = z_matrix.var(axis=1).sort_values(ascending=False)
        z_matrix = z_matrix.loc[kpi_variance.index]

        z_matrix.columns = [tactical_names.get(int(col), {}).get('name', f'C{col}') for col in z_matrix.columns]
        
        # Use readable KPI names
        z_matrix.index = [config.KPI_READABLE_NAMES.get(kpi, kpi) for kpi in z_matrix.index]

        # Custom colorscale: more contrast and better visibility
        custom_colorscale = [
            [0, '#1e3a5f'],      # Dark blue for very low
            [0.25, '#4a90c2'],   # Medium blue for low
            [0.5, '#f5f5f5'],    # Light gray for average
            [0.75, '#e07b54'],   # Orange for high
            [1, '#b91c1c']       # Dark red for very high
        ]

        fig = go.Figure(data=go.Heatmap(
            z=z_matrix.values,
            x=z_matrix.columns,
            y=z_matrix.index,
            colorscale=custom_colorscale,
            zmid=0,
            zmin=-3,
            zmax=3,
            colorbar=dict(
                title=dict(text='Z-Score', font=dict(color='#fff', size=14, family='Arial')),
                tickfont=dict(color='#fff', size=12, family='Arial'),
                thickness=20,
                len=0.9,
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                ticktext=['-3', '-2', '-1', '0', '+1', '+2', '+3'],
                outlinewidth=0
            ),
            hovertemplate=(
                "<b style='font-size:14px'>%{y}</b><br>"
                "<span style='color:#888'>Cluster:</span> <b>%{x}</b><br>"
                "<span style='color:#888'>Z-Score:</span> <b>%{z:+.2f}</b>"
                "<extra></extra>"
            ),
            text=z_matrix.values.round(2),
            texttemplate="%{text:+.2f}",
            textfont=dict(
                size=14, 
                family='Arial Black',
                color='white'
            )
        ))

        fig.update_layout(
            height=max(500, 50 * len(kpis)),
            margin=dict(t=40, b=140, l=300, r=140),
            xaxis=dict(
                title=dict(
                    text='<b>Cluster</b>', 
                    font=dict(size=15, color='#e0e0e0', family='Arial')
                ),
                tickangle=-30, 
                tickfont=dict(size=14, color='#f0f0f0', family='Arial'),
                side='bottom',
                showticklabels=True
            ),
            yaxis=dict(
                title=dict(
                    text='<b>KPI</b>', 
                    font=dict(size=15, color='#e0e0e0', family='Arial')
                ),
                tickfont=dict(size=13, color='#f0f0f0', family='Arial'),
                autorange='reversed',
                showticklabels=True
            ),
            paper_bgcolor='rgba(26, 26, 46, 1)',
            plot_bgcolor='rgba(26, 26, 46, 1)',
            font=dict(color='#fff', family='Arial'),
            hoverlabel=dict(
                bgcolor='rgba(30, 30, 50, 0.95)',
                bordercolor='#f093fb',
                font=dict(size=13, color='#fff', family='Arial')
            )
        )
        
        # Ensure labels are visible in export
        fig.update_xaxes(
            tickmode='array', 
            tickvals=list(range(len(z_matrix.columns))), 
            ticktext=[f"<b>{col}</b>" for col in z_matrix.columns]
        )
        fig.update_yaxes(
            tickmode='array', 
            tickvals=list(range(len(z_matrix.index))), 
            ticktext=[f"{idx}" for idx in z_matrix.index]
        )

        return fig

    heatmap_fig = create_z_score_heatmap(z_scores_df, tactical_names, selected_heatmap_kpis, st.session_state.theme)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    add_download_button(heatmap_fig, 'zscore_heatmap', 'Z_Score_Heatmap')
else:
    st.warning("⚠️ Select at least 3 KPIs for heatmap")

# ============================================================================
# CLUSTER DETAILS - Enhanced
# ============================================================================

st.markdown("---")
st.markdown('<div class="section-title">🎯 Cluster Profiles</div>', unsafe_allow_html=True)

cluster_ids = sorted(clustered_df['cluster_label'].unique())

# Cluster Overview Cards
st.markdown("#### Quick Overview")

cols = st.columns(len(cluster_ids))

cluster_colors_list = ['#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']

for idx, cluster_id in enumerate(cluster_ids):
    cluster_info = tactical_names.get(cluster_id, {})
    cluster_name = cluster_info.get('name', f'Cluster {cluster_id}')
    cluster_players = get_cluster_players(selected_position, cluster_id)
    player_count = len(cluster_players)
    
    # Check if silhouette_score exists and has valid values
    if 'silhouette_score' in cluster_players.columns and cluster_players['silhouette_score'].notna().any():
        avg_sil = cluster_players['silhouette_score'].mean()
        sil_display = f"{avg_sil:.2f}"
    else:
        sil_display = "N/A"
    
    avg_minutes = cluster_players['minutes_played'].mean() if 'minutes_played' in cluster_players.columns else 0
    
    color = cluster_colors_list[idx % len(cluster_colors_list)]
    
    with cols[idx]:
        st.markdown(f"""
        <div class="cluster-card" style="border-left-color: {color};">
            <div class="cluster-header">
                <div class="cluster-icon" style="background: {color};">C{cluster_id}</div>
                <div>
                    <div class="cluster-name">{cluster_name}</div>
                </div>
            </div>
            <div class="cluster-stats">
                <div class="cluster-stat">
                    <div class="cluster-stat-value">{player_count}</div>
                    <div class="cluster-stat-label">Players</div>
                </div>
                <div class="cluster-stat">
                    <div class="cluster-stat-value">{sil_display}</div>
                    <div class="cluster-stat-label">Silhouette</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# DETAILED CLUSTER ANALYSIS
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Detailed Analysis")

# Cluster Selector with Tabs
selected_cluster = st.radio(
    "Select Cluster",
    options=cluster_ids,
    format_func=lambda x: f"{tactical_names.get(x, {}).get('name', f'Cluster {x}')}",
    horizontal=True,
    label_visibility="collapsed"
)

cluster_info = tactical_names.get(selected_cluster, {})
cluster_players = get_cluster_players(selected_position, selected_cluster)
cluster_name = cluster_info.get('name', f'Cluster {selected_cluster}')

# Cluster Info Display
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="description-box">
        <div class="description-title">📝 Cluster Description</div>
        <div class="description-text">
            {cluster_info.get('justification', 'This cluster groups players with similar performance characteristics based on the selected KPIs.')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key KPIs
    if 'top_3_kpis' in cluster_info:
        st.markdown("**🔑 Characteristic KPIs:**")
        kpi_html = ""
        for kpi_item in cluster_info['top_3_kpis']:
            # Handle both dict format {"kpi": "...", "z_score": ...} and string format
            if isinstance(kpi_item, dict):
                kpi_name = kpi_item.get('kpi', '')
                z_score = kpi_item.get('z_score', 0)
                readable = config.KPI_READABLE_NAMES.get(kpi_name, kpi_name)
                kpi_html += f'<span class="kpi-tag">{readable} (z={z_score:+.2f})</span>'
            else:
                readable = config.KPI_READABLE_NAMES.get(kpi_item, kpi_item)
                kpi_html += f'<span class="kpi-tag">{readable}</span>'
        st.markdown(kpi_html, unsafe_allow_html=True)

with col2:
    # Cluster Stats
    has_silhouette = 'silhouette_score' in cluster_players.columns and cluster_players['silhouette_score'].notna().any()
    avg_sil = cluster_players['silhouette_score'].mean() if has_silhouette else None
    avg_minutes = cluster_players['minutes_played'].mean() if 'minutes_played' in cluster_players.columns else 0
    
    st.metric("👥 Players", len(cluster_players))
    st.metric("📊 Avg Silhouette", f"{avg_sil:.3f}" if avg_sil is not None else "N/A")
    st.metric("⏱️ Avg Minutes", f"{avg_minutes:.0f}")

# ============================================================================
# PLAYER TABLE - Enhanced
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div class="player-table-container">
    <div class="player-table-header">
        <div style="font-size: 1.1rem; font-weight: 600; color: #fff;">
            👥 {cluster_name} - Player List
        </div>
        <div style="color: #888; font-size: 0.85rem;">
            {len(cluster_players)} players
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sort Options
col1, col2 = st.columns([2, 1])

with col1:
    sort_by = st.selectbox(
        "Sort by",
        options=['player_name', 'minutes_played', 'silhouette_score'],
        format_func=lambda x: {'player_name': '🔤 Name', 'minutes_played': '⏱️ Minutes', 'silhouette_score': '📊 Silhouette'}[x],
        label_visibility="collapsed"
    )

with col2:
    sort_asc = st.checkbox("Ascending", value=(sort_by == 'player_name'))

# Sort and Display
cluster_players_sorted = cluster_players.sort_values(by=sort_by, ascending=sort_asc)

display_columns = ['player_name', 'team', 'minutes_played', 'matches_played', 'silhouette_score']
display_columns = [col for col in display_columns if col in cluster_players_sorted.columns]

player_table = cluster_players_sorted[display_columns].copy()
player_table = player_table.rename(columns={
    'player_name': '👤 Player',
    'team': '🏟️ Team',
    'minutes_played': '⏱️ Minutes',
    'matches_played': '🎮 Matches',
    'silhouette_score': '📊 Silhouette'
})

if '⏱️ Minutes' in player_table.columns:
    player_table['⏱️ Minutes'] = player_table['⏱️ Minutes'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
if '📊 Silhouette' in player_table.columns:
    player_table['📊 Silhouette'] = player_table['📊 Silhouette'].apply(lambda x: f"{x:.3f}" if pd.notna(x) and x != 0 else "N/A")

st.dataframe(
    player_table, 
    use_container_width=True, 
    hide_index=True, 
    height=min(450, 38 * len(player_table) + 40)
)

# ============================================================================
# CLUSTER COMPARISON TABLE
# ============================================================================

st.markdown("---")
st.markdown('<div class="section-title">📋 All Clusters Summary</div>', unsafe_allow_html=True)

summary_data = []

for cluster_id in cluster_ids:
    c_info = tactical_names.get(cluster_id, {})
    c_players = get_cluster_players(selected_position, cluster_id)
    
    has_sil = 'silhouette_score' in c_players.columns and c_players['silhouette_score'].notna().any()
    avg_sil = c_players['silhouette_score'].mean() if has_sil else None
    min_sil = c_players['silhouette_score'].min() if has_sil else None
    avg_min = c_players['minutes_played'].mean() if 'minutes_played' in c_players.columns else 0
    
    summary_data.append({
        'Cluster': f"C{cluster_id}",
        'Name': c_info.get('name', f'Cluster {cluster_id}'),
        'Players': len(c_players),
        'Avg Silhouette': f"{avg_sil:.3f}" if avg_sil is not None else "N/A",
        'Min Silhouette': f"{min_sil:.3f}" if min_sil is not None else "N/A",
        'Avg Minutes': f"{avg_min:.0f}"
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

with st.expander("ℹ️ How to Interpret Clusters"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 PCA Scatter Plot
        - **Purpose:** Dimensionality reduction for 2D visualization
        - **Interpretation:** Well-separated clusters = successful clustering
        - **Tip:** Overlapping points may indicate similar players
        """)
        
    with col2:
        st.markdown("""
        ### 🌡️ Z-Score Heatmap
        - **Red cells:** Above average performance
        - **Blue cells:** Below average performance
        - **Interpretation:** Shows each cluster's strengths/weaknesses
        """)
    
    st.markdown("""
    ### 📊 Silhouette Score
    - **Range:** -1 to 1
    - **High value (> 0.5):** Player fits well in cluster
    - **Low value (< 0):** Player may be misclassified
    - **Interpretation:** Measures how similar a player is to their own cluster vs other clusters
    """)

st.caption("**Methodology:** K-Means Clustering + PCA Visualization + Silhouette Analysis")
