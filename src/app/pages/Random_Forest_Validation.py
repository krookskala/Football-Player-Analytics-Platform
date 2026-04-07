"""
Feature Importance Dashboard - Modern
--------------------------------------
RF feature importance vs F-statistics comparison with validation metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
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
    load_rf_results,
    load_f_statistics,
    load_all_positions_summary
)

# ============================================================================
# MODERN CSS STYLES
# ============================================================================

st.markdown("""
<style>
/* Page Header */
.page-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #11998e, #38ef7d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-subtitle {
    color: #a8dadc;
    font-size: 0.95rem;
}

/* Validation Card */
.validation-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.validation-card.success {
    border-color: rgba(46, 204, 113, 0.5);
    background: linear-gradient(145deg, rgba(46, 204, 113, 0.1) 0%, rgba(46, 204, 113, 0.02) 100%);
}

.validation-card.warning {
    border-color: rgba(241, 196, 15, 0.5);
    background: linear-gradient(145deg, rgba(241, 196, 15, 0.1) 0%, rgba(241, 196, 15, 0.02) 100%);
}

.validation-card.error {
    border-color: rgba(231, 76, 60, 0.5);
    background: linear-gradient(145deg, rgba(231, 76, 60, 0.1) 0%, rgba(231, 76, 60, 0.02) 100%);
}

.validation-value {
    font-size: 2rem;
    font-weight: 700;
}

.validation-value.success { color: #2ecc71; }
.validation-value.warning { color: #f1c40f; }
.validation-value.error { color: #e74c3c; }

.validation-label {
    color: #888;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* Status Badge */
.status-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.85rem;
}

.status-validated {
    background: linear-gradient(90deg, #11998e, #38ef7d);
    color: #fff;
}

.status-weak {
    background: linear-gradient(90deg, #f39c12, #e74c3c);
    color: #fff;
}

.status-failed {
    background: #e74c3c;
    color: #fff;
}

/* Section Title */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(17, 153, 142, 0.3);
}

/* Info Box */
.info-box {
    background: rgba(17, 153, 142, 0.1);
    border-left: 3px solid #11998e;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    font-size: 0.9rem;
}

/* Position Summary Card */
.position-summary {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}

.position-name {
    font-weight: 600;
    color: #fff;
    font-size: 1rem;
}

.position-stats {
    color: #888;
    font-size: 0.85rem;
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
    <div class="page-title">🔬 Feature Importance Dashboard</div>
    <div class="page-subtitle">
        Random Forest validation & feature analysis for clustering quality assessment
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# POSITION INFO
# ============================================================================

selected_position = st.session_state.selected_position
pos_info = config.POSITIONS[selected_position]

st.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.5rem;">
    <span style="font-size: 2rem;">{pos_info['icon']}</span>
    <div>
        <div style="font-size: 1.3rem; font-weight: 600;">{pos_info['name']}</div>
        <div style="color: #888; font-size: 0.85rem;">{pos_info['n_players']} players · {pos_info['n_clusters']} clusters</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD RF RESULTS
# ============================================================================

try:
    rf_results = load_rf_results(selected_position)
except FileNotFoundError:
    st.error(f"❌ RF results not found for {selected_position}")
    st.stop()

try:
    f_stats = load_f_statistics(selected_position)
    f_stats_available = True
except FileNotFoundError:
    st.warning("⚠️ F-statistics not available")
    f_stats_available = False

# ============================================================================
# VALIDATION METRICS
# ============================================================================

st.markdown('<div class="section-title">📊 Validation Metrics</div>', unsafe_allow_html=True)

cv_accuracy = rf_results.get('cv_accuracy', 0)
cv_std = rf_results.get('cv_std', 0)
spearman_rho = rf_results.get('spearman_rho', 0)
validation_status = rf_results.get('validation_status', 'UNKNOWN')

# Determine card classes
acc_class = 'success' if cv_accuracy > 0.8 else ('warning' if cv_accuracy > 0.7 else 'error')
rho_class = 'success' if spearman_rho > 0.5 else ('warning' if spearman_rho > 0.3 else 'error')
status_class = 'status-validated' if validation_status == 'VALIDATED' else ('status-weak' if validation_status == 'WEAK' else 'status-failed')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="validation-card {acc_class}">
        <div class="validation-value {acc_class}">{cv_accuracy:.1%}</div>
        <div class="validation-label">CV Accuracy</div>
        <div style="color: #666; font-size: 0.75rem; margin-top: 0.3rem;">±{cv_std:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="validation-card {rho_class}">
        <div class="validation-value {rho_class}">{spearman_rho:.3f}</div>
        <div class="validation-label">Spearman ρ</div>
        <div style="color: #666; font-size: 0.75rem; margin-top: 0.3rem;">RF vs F-stats</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status_emoji = '✅' if validation_status == 'VALIDATED' else ('⚠️' if validation_status == 'WEAK' else '❌')
    st.markdown(f"""
    <div class="validation-card">
        <div style="font-size: 2rem;">{status_emoji}</div>
        <div class="validation-label">Status</div>
        <div style="margin-top: 0.5rem;">
            <span class="{status_class}">{validation_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Status Message
if validation_status == 'VALIDATED':
    st.success("✅ **High Quality Clustering**: RF and F-statistics show strong correlation")
elif validation_status == 'WEAK':
    st.warning("⚠️ **Moderate Quality**: Correlation acceptable but could be improved")
else:
    st.error("❌ **Low Quality**: Review clustering parameters")

# ============================================================================
# FEATURE IMPORTANCE BAR CHART
# ============================================================================

st.markdown("---")
st.markdown('<div class="section-title">🌲 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

feature_importance = rf_results.get('feature_importance', {})

if feature_importance:
    importance_df = pd.DataFrame([
        {'kpi': kpi, 'importance': imp}
        for kpi, imp in feature_importance.items()
    ]).sort_values('importance', ascending=False)

    max_kpis = len(importance_df)
    min_kpis = min(3, max_kpis)
    default_kpis = min(12, max_kpis)
    
    if max_kpis > min_kpis:
        top_n = st.slider("Number of KPIs:", min_kpis, max_kpis, default_kpis)
    else:
        top_n = max_kpis
        st.caption(f"Showing all {max_kpis} KPIs")
    importance_top = importance_df.head(top_n)

    fig = px.bar(
        importance_top,
        x='importance',
        y='kpi',
        orientation='h',
        color='importance',
        color_continuous_scale=['#0f3460', '#11998e', '#38ef7d'],
        text='importance'
    )

    fig.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside',
        marker_line_width=0
    )

    fig.update_layout(
        height=max(350, 28 * len(importance_top)),
        showlegend=False,
        margin=dict(t=20, b=40, l=180, r=80),
        xaxis=dict(title='Importance Score', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='', autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fff')
    )

    st.plotly_chart(fig, use_container_width=True)
    add_download_button(fig, 'rf_importance', 'Feature_Importance')

    with st.expander("📊 Full Importance Table"):
        table_df = importance_df.copy()
        table_df['rank'] = range(1, len(table_df) + 1)
        table_df['importance'] = table_df['importance'].apply(lambda x: f"{x:.4f}")
        st.dataframe(table_df[['rank', 'kpi', 'importance']], hide_index=True, use_container_width=True)
else:
    st.warning("⚠️ Feature importance data not available")

# ============================================================================
# RF vs F-STATISTICS CORRELATION
# ============================================================================

if f_stats_available and feature_importance:
    st.markdown("---")
    st.markdown('<div class="section-title">📈 RF vs F-Statistics Correlation</div>', unsafe_allow_html=True)

    comparison_data = []
    for kpi, rf_imp in feature_importance.items():
        f_stat_data = f_stats.get(kpi, {})
        if isinstance(f_stat_data, dict):
            f_value = f_stat_data.get('f_statistic', 0)
        else:
            f_value = float(f_stat_data) if f_stat_data else 0

        comparison_data.append({
            'kpi': kpi,
            'rf_importance': rf_imp,
            'f_statistic': f_value
        })

    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        comparison_df['rf_normalized'] = comparison_df['rf_importance'] / comparison_df['rf_importance'].max()
        comparison_df['f_normalized'] = comparison_df['f_statistic'] / comparison_df['f_statistic'].max()

        fig = px.scatter(
            comparison_df,
            x='f_normalized',
            y='rf_normalized',
            hover_data=['kpi', 'rf_importance', 'f_statistic'],
            color='rf_importance',
            color_continuous_scale=['#0f3460', '#11998e', '#38ef7d'],
            size='rf_importance',
            size_max=18
        )

        # Add regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            comparison_df['f_normalized'], comparison_df['rf_normalized']
        )

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[intercept, slope + intercept],
            mode='lines',
            name=f'Linear Fit (R²={r_value**2:.2f})',
            line=dict(color='#f39c12', width=2, dash='dash')
        ))

        fig.update_layout(
            height=500,
            title=dict(text=f"Spearman ρ = {spearman_rho:.3f}", font=dict(size=16, color='#fff')),
            margin=dict(t=60, b=40, l=50, r=50),
            xaxis=dict(title='F-Statistic (normalized)', range=[-0.05, 1.05], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='RF Importance (normalized)', range=[-0.05, 1.05], showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fff'),
            showlegend=True
        )

        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>RF: %{customdata[1]:.4f}<br>F-stat: %{customdata[2]:.2f}<extra></extra>",
            selector=dict(type='scatter', mode='markers')
        )

        st.plotly_chart(fig, use_container_width=True)
        add_download_button(fig, 'rf_vs_fstat', 'Correlation_Plot')

# ============================================================================
# ALL POSITIONS SUMMARY
# ============================================================================

st.markdown("---")
st.markdown('<div class="section-title">🌍 All Positions Summary</div>', unsafe_allow_html=True)

summary_data = load_all_positions_summary()

cols = st.columns(3)
for idx, position_key in enumerate(config.POSITION_KEYS):
    pos_summary = summary_data.get(position_key)
    if not pos_summary:
        continue
        
    pos_config = config.POSITIONS.get(position_key, {})
    status = pos_summary.get('validation_status', 'N/A')
    status_emoji = '✅' if status == 'VALIDATED' else ('⚠️' if status == 'WEAK' else '❓')
    
    with cols[idx % 3]:
        st.markdown(f"""
        <div class="position-summary">
            <div class="position-name">{pos_config.get('icon', '')} {pos_summary.get('position_name', position_key)}</div>
            <div class="position-stats">
                👥 {pos_summary.get('n_players', 0)} players · 
                🎯 {pos_summary.get('n_clusters', 0)} clusters<br>
                📊 Accuracy: {pos_summary.get('rf_accuracy', 0):.1%} · 
                ρ: {pos_summary.get('spearman_rho', 0):.2f} {status_emoji}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Summary Stats
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

validated_count = sum(1 for d in summary_data.values() if d and d.get('validation_status') == 'VALIDATED')
accuracies = [d['rf_accuracy'] for d in summary_data.values() if d and d.get('rf_accuracy')]
spearmans = [d['spearman_rho'] for d in summary_data.values() if d and d.get('spearman_rho')]

col1.metric("Validated Positions", f"{validated_count}/6")
col2.metric("Mean RF Accuracy", f"{sum(accuracies)/len(accuracies):.1%}" if accuracies else "N/A")
col3.metric("Mean Spearman ρ", f"{sum(spearmans)/len(spearmans):.3f}" if spearmans else "N/A")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

with st.expander("ℹ️ Understanding Feature Importance"):
    st.markdown("""
    ### Interpretation Guide
    
    **RF Importance:** Contribution of each KPI to cluster prediction
    - High = discriminative (distinguishes clusters well)
    - Low = non-discriminative
    
    **F-Statistics:** Significance of differences between clusters (ANOVA)
    - High F-value = large between-cluster variance
    
    **Spearman ρ:** Rank correlation between RF and F-stats
    - ρ > 0.7 = very high agreement
    - ρ > 0.5 = validated
    - ρ < 0.5 = weak/failed
    
    **Validation Status:**
    - ✅ VALIDATED: ρ > 0.5 AND accuracy > 70%
    - ⚠️ WEAK: One criterion not met
    - ❌ FAILED: Both criteria not met
    """)

st.caption("**Methodology:** Random Forest Classifier + 5-Fold CV + Spearman Rank Correlation")
