"""
Home Page - Modern Dashboard
----------------------------
Professional analytics dashboard with hero section and metric cards.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import config
from data_loader import load_all_positions_summary
from demo_modes import display_demo_buttons, initialize_demo_state
from utils.ui import (
    display_breadcrumb,
    inject_presentation_mode_css,
    display_contextual_help
)

# ============================================================================
# CUSTOM CSS FOR MODERN DESIGN
# ============================================================================

st.markdown("""
<style>
/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 1.5rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(230, 57, 70, 0.1) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e63946, #f4a261, #2a9d8f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    font-size: 1rem;
    color: #a8dadc;
    margin-bottom: 1rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(230, 57, 70, 0.2);
    border: 1px solid #e63946;
    color: #e63946;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: rgba(230, 57, 70, 0.5);
    box-shadow: 0 10px 40px rgba(230, 57, 70, 0.2);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e63946, #f4a261);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.9rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

/* Position Cards */
.position-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    cursor: pointer;
}

.position-card:hover {
    border-color: #2a9d8f;
    background: linear-gradient(145deg, rgba(42,157,143,0.15) 0%, rgba(42,157,143,0.05) 100%);
}

.position-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.position-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff;
}

.position-stats {
    font-size: 0.8rem;
    color: #888;
}

/* Section Headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(230, 57, 70, 0.3);
}

/* Feature Cards */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.feature-card h4 {
    color: #e63946;
    margin-bottom: 0.5rem;
}

.feature-card p {
    flex-grow: 1;
}

/* Status Badge */
.status-validated {
    background: rgba(42, 157, 143, 0.2);
    color: #2a9d8f;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.status-weak {
    background: rgba(244, 162, 97, 0.2);
    color: #f4a261;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Animated gradient border */
@keyframes gradient-border {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradient-border {
    background: linear-gradient(90deg, #e63946, #f4a261, #2a9d8f, #e63946);
    background-size: 300% 300%;
    animation: gradient-border 5s ease infinite;
    padding: 2px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

if 'presentation_mode' not in st.session_state:
    st.session_state.presentation_mode = False

st.sidebar.markdown("---")
presentation_mode = st.sidebar.checkbox(
    "🎤 Presentation Mode",
    value=st.session_state.presentation_mode,
    help="Enable large fonts for thesis defense"
)
st.session_state.presentation_mode = presentation_mode

inject_presentation_mode_css()
initialize_demo_state()

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">⚽ Football Player Archetype Analytics</div>
    <div class="hero-subtitle">
        Machine Learning-Powered Tactical Role Identification | StatsBomb Open Data
    </div>
    <div>
        <span class="hero-badge">🏆 FIFA World Cup 2022</span>
        <span class="hero-badge">🤖 K-Means Clustering</span>
        <span class="hero-badge">🌲 Random Forest Validation</span>
        <span class="hero-badge">📊 358 Players</span>
        <span class="hero-badge">🎯 16 Archetypes</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

summary_data = load_all_positions_summary()

total_players = sum([s['n_players'] for s in summary_data.values() if s])
total_clusters = sum([s['n_clusters'] for s in summary_data.values() if s])
positions_count = len([s for s in summary_data.values() if s])

validated_positions = [
    s for s in summary_data.values()
    if s and s.get('validation_status') == 'VALIDATED'
]

mean_rf_accuracy = (
    sum([s['rf_accuracy'] for s in validated_positions if s.get('rf_accuracy')]) /
    len(validated_positions) if validated_positions else 0
)

silhouette_values = [s['mean_silhouette'] for s in summary_data.values() if s and s.get('mean_silhouette')]
mean_silhouette = sum(silhouette_values) / len(silhouette_values) if silhouette_values else 0

# ============================================================================
# METRIC CARDS
# ============================================================================

st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">👥</div>
        <div class="metric-value">{total_players}</div>
        <div class="metric-label">Total Players</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🎯</div>
        <div class="metric-value">{total_clusters}</div>
        <div class="metric-label">Player Archetypes</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🎖️</div>
        <div class="metric-value">{mean_rf_accuracy:.0%}</div>
        <div class="metric-label">RF Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-value">{mean_silhouette:.2f}</div>
        <div class="metric-label">Silhouette Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# POSITION CARDS
# ============================================================================

st.markdown('<div class="section-header">🏟️ Position Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

position_icons = {
    'midfielder': '⚙️',
    'center_back': '🛡️',
    'full_back': '🏃',
    'winger': '⚡',
    'forward': '⚽',
    'goalkeeper': '🧤'
}

position_colors = {
    'midfielder': '#3498db',
    'center_back': '#e74c3c',
    'full_back': '#2ecc71',
    'winger': '#f39c12',
    'forward': '#9b59b6',
    'goalkeeper': '#1abc9c'
}

positions_list = list(config.POSITIONS.keys())

for idx, pos_key in enumerate(positions_list):
    pos_data = summary_data.get(pos_key, {})
    pos_info = config.POSITIONS.get(pos_key, {})
    
    if not pos_data:
        continue
    
    col = [col1, col2, col3][idx % 3]
    
    with col:
        status = pos_data.get('validation_status', 'N/A')
        status_class = 'status-validated' if status == 'VALIDATED' else 'status-weak'
        status_text = '✓ Validated' if status == 'VALIDATED' else '⚠ Review'
        
        silhouette = pos_data.get('mean_silhouette', 0)
        sil_display = f"{silhouette:.3f}" if silhouette else "N/A"
        
        st.markdown(f"""
        <div class="position-card">
            <div class="position-icon">{position_icons.get(pos_key, '📊')}</div>
            <div class="position-name">{pos_info.get('name', pos_key)}</div>
            <div class="position-stats">
                {pos_data.get('n_players', 0)} players · {pos_data.get('n_clusters', 0)} clusters · Sil: {sil_display}
            </div>
            <div style="margin-top: 0.5rem;">
                <span class="{status_class}">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# QUICK ACTIONS
# ============================================================================

st.markdown('<div class="section-header">🚀 Quick Actions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.5rem; min-height: 180px;">
            <h4 style="color: #e63946; margin-bottom: 0.8rem;">🎯 Player Scouting</h4>
            <p style="color: #888; font-size: 0.9rem; margin-bottom: 1rem;">
                Compare players with pizza charts and percentile rankings. Head-to-head analysis available.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open Scouting Tool", use_container_width=True, type="primary", key="btn_scouting")

with col2:
    with st.container():
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.5rem; min-height: 180px;">
            <h4 style="color: #e63946; margin-bottom: 0.8rem;">📊 Cluster Analysis</h4>
            <p style="color: #888; font-size: 0.9rem; margin-bottom: 1rem;">
                Explore PCA scatter plots, z-score heatmaps, and player clusters.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open Clustering", use_container_width=True, type="primary", key="btn_clustering")

with col3:
    with st.container():
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.5rem; min-height: 180px;">
            <h4 style="color: #e63946; margin-bottom: 0.8rem;">🔬 RF Validation</h4>
            <p style="color: #888; font-size: 0.9rem; margin-bottom: 1rem;">
                Random Forest feature importance and validation metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Open Validation", use_container_width=True, type="primary", key="btn_validation")

# Handle button clicks
if st.session_state.get("btn_scouting"):
    st.switch_page("pages/Scouting_Tool.py")
if st.session_state.get("btn_clustering"):
    st.switch_page("pages/K_Means_Clustering_Analysis.py")
if st.session_state.get("btn_validation"):
    st.switch_page("pages/Random_Forest_Validation.py")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# DATA STATUS TABLE
# ============================================================================

with st.expander("📋 Detailed Position Statistics", expanded=False):
    status_table_data = []
    
    for position_key in config.POSITION_KEYS:
        pos_summary = summary_data.get(position_key)
        
        if pos_summary:
            status_table_data.append({
                'Position': pos_summary['position_name'],
                'Players': pos_summary['n_players'],
                'Clusters': pos_summary['n_clusters'],
                'Silhouette': f"{pos_summary['mean_silhouette']:.3f}" if pos_summary.get('mean_silhouette') else 'N/A',
                'RF Accuracy': f"{pos_summary['rf_accuracy']:.1%}" if pos_summary.get('rf_accuracy') else 'N/A',
                'Spearman ρ': f"{pos_summary['spearman_rho']:.3f}" if pos_summary.get('spearman_rho') else 'N/A',
                'Status': pos_summary.get('validation_status', 'N/A')
            })
    
    status_df = pd.DataFrame(status_table_data)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

# ============================================================================
# METHODOLOGY (Collapsible)
# ============================================================================

with st.expander("📚 Methodology Overview", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 Clustering Pipeline")
        st.markdown("""
        1. **Data Source**: StatsBomb Open Data (World Cup 2022)
        2. **Positions**: 6 tactical categories (358 players)
        3. **KPIs**: 5-7 position-specific metrics (39 total)
        4. **Normalization**: Z-score standardization
        5. **Algorithm**: K-Means (k=2 or k=4 per position)
        6. **Optimal K**: Elbow, Silhouette, Davies-Bouldin, CH
        """)
    
    with col2:
        st.markdown("#### 🌲 Random Forest Validation")
        st.markdown("""
        1. **Model**: Random Forest (100 trees, Gini)
        2. **CV**: 5-fold stratified cross-validation
        3. **Mean Accuracy**: 93.3% (outfield positions)
        4. **Spearman ρ**: 0.872 (feature correlation)
        5. **Archetypes**: 16 distinct player profiles
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("📊 **Data Source**")
    st.caption("StatsBomb Open Data")

with col2:
    st.caption("🎓 **Project Type**")
    st.caption("Undergraduate Thesis")

with col3:
    st.caption("🛠️ **Tech Stack**")
    st.caption("Python · Streamlit · Plotly")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "⚽ Football Player Analytics Platform | FIFA World Cup 2022 | "
    "K-Means Clustering + Random Forest Validation"
    "</div>",
    unsafe_allow_html=True
)
