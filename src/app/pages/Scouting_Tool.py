"""
Player Scouting & Comparison
----------------------------
Modern scouting tool with pizza charts and head-to-head comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
import config
from data_loader import (
    load_clustered_data,
    load_tactical_names,
    get_position_player_names,
    compute_percentiles
)
from utils.charts import create_pizza_chart, create_comparison_chart, add_download_button
from utils.ui import (
    display_breadcrumb,
    inject_presentation_mode_css,
    display_contextual_help,
    display_loading_skeleton
)
from utils.pdf_report import generate_player_comparison_pdf

# ============================================================================
# MODERN CSS STYLES
# ============================================================================

st.markdown("""
<style>
/* Page Header */
.page-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b68ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.page-subtitle {
    color: #a8dadc;
    font-size: 0.95rem;
}

/* Player Card */
.player-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.player-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
}

.player-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
    margin: 0.8rem 0 0.3rem 0;
}

.player-team {
    color: #888;
    font-size: 0.9rem;
}

.player-cluster {
    display: inline-block;
    background: linear-gradient(90deg, #00d4ff, #7b68ee);
    color: #fff;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.8rem;
}

/* VS Badge */
.vs-badge {
    background: linear-gradient(135deg, #e63946, #ff6b6b);
    color: #fff;
    font-size: 1.5rem;
    font-weight: 800;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
    box-shadow: 0 4px 15px rgba(230, 57, 70, 0.4);
}

/* Stat Row */
.stat-row {
    display: flex;
    align-items: center;
    padding: 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    min-width: 80px;
    text-align: center;
}

.stat-label {
    flex: 1;
    text-align: center;
    color: #888;
    font-size: 0.9rem;
}

.stat-winner {
    color: #2ecc71;
}

.stat-loser {
    color: #e74c3c;
}

/* Position Pills */
.position-pills {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}

.position-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 0.5rem 1rem;
    border-radius: 25px;
    color: #fff;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.position-pill:hover {
    background: rgba(0, 212, 255, 0.2);
    border-color: #00d4ff;
}

.position-pill.active {
    background: linear-gradient(90deg, #00d4ff, #7b68ee);
    border-color: transparent;
}

/* Chart Container */
.chart-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem 0;
}

/* Section Title */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #fff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(0, 212, 255, 0.3);
}

/* Info Box */
.info-box {
    background: rgba(0, 212, 255, 0.1);
    border-left: 3px solid #00d4ff;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

inject_presentation_mode_css()

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'selected_position' not in st.session_state:
    st.session_state.selected_position = 'midfielder'

# Header
st.markdown("""
<div class="page-header">
    <div class="page-title">🎯 Player Scouting & Comparison</div>
    <div class="page-subtitle">
        Analyze players' percentile ranks and discover strengths with pizza charts
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# POSITION SELECTION
# ============================================================================

positions = list(config.POSITIONS.keys())
display_names = [config.POSITIONS[p]['name'] for p in positions]

current_pos = st.session_state.get('selected_position', 'midfielder')
try:
    current_index = positions.index(current_pos)
except ValueError:
    current_index = 0

selected_display = st.radio(
    "Position Filter",
    options=display_names,
    index=current_index,
    horizontal=True,
    label_visibility="collapsed"
)

selected_position = positions[display_names.index(selected_display)]
st.session_state.selected_position = selected_position
pos_info = config.POSITIONS[selected_position]

# Position Badge
st.markdown(f"""
<div style="margin-bottom: 1rem;">
    <span style="font-size: 1.5rem;">{pos_info['icon']}</span>
    <span style="font-size: 1.2rem; font-weight: 600; margin-left: 0.5rem;">{pos_info['name']}</span>
    <span style="color: #888; margin-left: 0.5rem;">({pos_info['n_players']} players)</span>
</div>
""", unsafe_allow_html=True)

# Load data
try:
    clustered_df = load_clustered_data(selected_position)
    percentiles_df = compute_percentiles(selected_position)
    tactical_names = load_tactical_names(selected_position)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================================================
# PLAYER SELECTION
# ============================================================================

player_names = get_position_player_names(selected_position)

if st.session_state.get('demo_mode') and st.session_state.get('selected_players'):
    default_players = [p for p in st.session_state.selected_players if p in player_names]
else:
    default_players = []

st.markdown('<div class="section-title">👤 Select Players</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    selected_players = st.multiselect(
        "Player Selection",
        options=player_names,
        default=default_players,
        max_selections=3,
        placeholder="🔍 Search player...",
        label_visibility="collapsed"
    )

with col2:
    st.caption("Select 1-3 players")

if not selected_players:
    st.markdown("""
    <div class="info-box">
        <strong>👆 How to use:</strong><br>
        • Select <strong>1 player</strong> for individual profile (Pizza Chart)<br>
        • Select <strong>2 players</strong> for head-to-head comparison<br>
        • Select <strong>3 players</strong> for multi-player comparison
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================================
# KPI SETTINGS
# ============================================================================

with st.expander("⚙️ KPI Settings", expanded=False):
    available_kpis = pos_info['kpis']
    selected_kpis = st.multiselect(
        "KPIs to display",
        options=available_kpis,
        default=available_kpis,
        format_func=lambda x: config.KPI_READABLE_NAMES.get(x, x)
    )

if len(selected_kpis) < 3:
    st.warning("⚠️ Select at least 3 KPIs")
    st.stop()

# ============================================================================
# PLAYER COMPARISON
# ============================================================================

st.markdown("---")

if selected_players:
    players_data = []
    for p in selected_players:
        row = clustered_df[clustered_df['player_name'] == p].iloc[0]
        players_data.append(row)

    # === HEAD-TO-HEAD MODE (2 Players) ===
    if len(selected_players) == 2:
        p1, p2 = players_data[0], players_data[1]
        
        # Player Cards
        c1, c2, c3 = st.columns([1, 0.5, 1])
        
        with c1:
            cluster_id_1 = int(p1.get('cluster_label', 0))
            cluster_name_1 = tactical_names.get(cluster_id_1, {}).get('name', f'Cluster {cluster_id_1}')
            team_1 = p1.get('team', p1.get('team_name', 'Unknown'))
            initials_1 = ''.join([n[0].upper() for n in p1['player_name'].split()[:2]])
            
            st.markdown(f"""
            <div class="player-card">
                <div style="width: 100px; height: 100px; border-radius: 50%; 
                     background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%);
                     display: flex; align-items: center; justify-content: center;
                     font-size: 2rem; font-weight: 700; color: white;
                     margin: 0 auto; box-shadow: 0 8px 25px rgba(30, 136, 229, 0.4);
                     border: 3px solid rgba(255,255,255,0.3);">
                    {initials_1}
                </div>
                <div class="player-name">{p1['player_name']}</div>
                <div class="player-team">🏟️ {team_1}</div>
                <div class="player-cluster">{cluster_name_1}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="width: 70px; height: 70px; border-radius: 50%;
                 background: linear-gradient(135deg, #e53935 0%, #ff6b6b 100%);
                 display: flex; align-items: center; justify-content: center;
                 margin: 0 auto; box-shadow: 0 6px 20px rgba(229, 57, 53, 0.5);
                 font-size: 1.3rem; font-weight: 800; color: white;">
                VS
            </div>
            """, unsafe_allow_html=True)

        with c3:
            cluster_id_2 = int(p2.get('cluster_label', 0))
            cluster_name_2 = tactical_names.get(cluster_id_2, {}).get('name', f'Cluster {cluster_id_2}')
            team_2 = p2.get('team', p2.get('team_name', 'Unknown'))
            initials_2 = ''.join([n[0].upper() for n in p2['player_name'].split()[:2]])
            
            st.markdown(f"""
            <div class="player-card">
                <div style="width: 100px; height: 100px; border-radius: 50%; 
                     background: linear-gradient(135deg, #e53935 0%, #ff6b6b 100%);
                     display: flex; align-items: center; justify-content: center;
                     font-size: 2rem; font-weight: 700; color: white;
                     margin: 0 auto; box-shadow: 0 8px 25px rgba(229, 57, 53, 0.4);
                     border: 3px solid rgba(255,255,255,0.3);">
                    {initials_2}
                </div>
                <div class="player-name">{p2['player_name']}</div>
                <div class="player-team">🏟️ {team_2}</div>
                <div class="player-cluster">{cluster_name_2}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison Chart
        st.markdown('<div class="section-title">📊 Radar Comparison</div>', unsafe_allow_html=True)

        chart_data = []
        for p in selected_players:
            p_row = clustered_df[clustered_df['player_name'] == p].iloc[0]
            p_pct = percentiles_df[percentiles_df['player_name'] == p].iloc[0]
            chart_data.append({
                'name': p,
                'percentiles': p_pct,
                'raw': p_row
            })

        fig = create_comparison_chart(
            players_data=chart_data,
            kpis=selected_kpis,
            theme=st.session_state.theme
        )

        col_L, col_M, col_R = st.columns([0.5, 3, 0.5])
        with col_M:
            st.plotly_chart(fig, use_container_width=True)
            add_download_button(fig, 'comparison_chart', 'H2H_Comparison')
            
        st.markdown("---")
        
        # Stats Comparison
        st.markdown('<div class="section-title">📈 Stats Comparison</div>', unsafe_allow_html=True)
        
        metrics_to_show = [
            ('matches_played', 'Matches', '🏟️'),
            ('minutes_played', 'Minutes', '⏱️')
        ] + [(k, config.KPI_READABLE_NAMES.get(k, k), '📊') for k in pos_info['kpis'][:8]]

        for metric_key, metric_name, icon in metrics_to_show:
            val1 = p1.get(metric_key, 0) or 0
            val2 = p2.get(metric_key, 0) or 0
            
            if metric_key in ['matches_played', 'minutes_played']:
                v1_str = f"{int(val1)}"
                v2_str = f"{int(val2)}"
            else:
                v1_str = f"{float(val1):.2f}"
                v2_str = f"{float(val2):.2f}"

            color1 = "#2ecc71" if val1 > val2 else ("#e74c3c" if val2 > val1 else "#f1c40f")
            color2 = "#2ecc71" if val2 > val1 else ("#e74c3c" if val1 > val2 else "#f1c40f")
            
            rc1, rc2, rc3 = st.columns([1, 2, 1])
            with rc1:
                st.markdown(f"<div style='text-align: center; color: {color1}; font-weight: bold; font-size: 1.1em;'>{v1_str}</div>", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"<div style='text-align: center; color: #888;'>{icon} {metric_name}</div>", unsafe_allow_html=True)
            with rc3:
                st.markdown(f"<div style='text-align: center; color: {color2}; font-weight: bold; font-size: 1.1em;'>{v2_str}</div>", unsafe_allow_html=True)
            
            st.divider()

    # === SINGLE/MULTI PLAYER MODE ===
    else:
        cols = st.columns(len(selected_players))
        colors = [
            ('linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%)', 'rgba(30, 136, 229, 0.4)'),
            ('linear-gradient(135deg, #e53935 0%, #ff6b6b 100%)', 'rgba(229, 57, 53, 0.4)'),
            ('linear-gradient(135deg, #43a047 0%, #66bb6a 100%)', 'rgba(67, 160, 71, 0.4)')
        ]
        
        for idx, player_name in enumerate(selected_players):
            with cols[idx]:
                player_row = clustered_df[clustered_df['player_name'] == player_name].iloc[0]
                cluster_id = int(player_row['cluster_label'])
                cluster_info = tactical_names.get(cluster_id, {})
                cluster_name = cluster_info.get('name', f'Cluster {cluster_id}')
                team = player_row.get('team', player_row.get('team_name', 'Unknown'))
                initials = ''.join([n[0].upper() for n in player_name.split()[:2]])
                
                bg_gradient, shadow_color = colors[idx % 3]
                
                st.markdown(f"""
                <div class="player-card">
                    <div style="width: 90px; height: 90px; border-radius: 50%; 
                         background: {bg_gradient};
                         display: flex; align-items: center; justify-content: center;
                         font-size: 1.8rem; font-weight: 700; color: white;
                         margin: 0 auto; box-shadow: 0 8px 25px {shadow_color};
                         border: 3px solid rgba(255,255,255,0.3);">
                        {initials}
                    </div>
                    <div class="player-name">{player_name}</div>
                    <div class="player-team">🏟️ {team}</div>
                    <div class="player-cluster">{cluster_name}</div>
                    <div style="margin-top: 0.8rem; color: #888; font-size: 0.85rem;">
                        🎮 {int(player_row.get('matches_played', 0))} matches · ⏱️ {int(player_row.get('minutes_played', 0))} min
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if len(selected_players) == 1:
            # Pizza Chart for single player
            player_name = selected_players[0]
            player_row = clustered_df[clustered_df['player_name'] == player_name].iloc[0]
            player_percentiles = percentiles_df[percentiles_df['player_name'] == player_name].iloc[0]
            
            st.markdown('<div class="section-title">🍕 Player Profile (Pizza Chart)</div>', unsafe_allow_html=True)
            
            col_chart, col_info = st.columns([2, 1])
            
            with col_chart:
                fig = create_pizza_chart(
                    player_name=player_name,
                    kpis=selected_kpis,
                    percentiles=player_percentiles,
                    raw_values=player_row,
                    theme=st.session_state.theme
                )
                st.plotly_chart(fig, use_container_width=True)
                add_download_button(fig, 'pizza_chart', 'Player_Profile')
                
            with col_info:
                st.markdown("#### 📊 Key Stats")
                if 'minutes_played' in player_row:
                    st.metric("Minutes", f"{player_row['minutes_played']:.0f}")
                if 'matches_played' in player_row:
                    st.metric("Matches", f"{player_row['matches_played']:.0f}")

                st.markdown("#### 🎨 Categories")
                for cat, color in config.CATEGORY_COLORS.items():
                    st.markdown(f"<span style='color:{color}'>■</span> {cat}", unsafe_allow_html=True)
        else:
            # Multi-player comparison
            st.markdown('<div class="section-title">⚔️ Multi-Player Comparison</div>', unsafe_allow_html=True)
            
            players_data = []
            for p in selected_players:
                p_row = clustered_df[clustered_df['player_name'] == p].iloc[0]
                p_pct = percentiles_df[percentiles_df['player_name'] == p].iloc[0]
                players_data.append({
                    'name': p,
                    'percentiles': p_pct,
                    'raw': p_row
                })
                
            fig = create_comparison_chart(
                players_data=players_data,
                kpis=selected_kpis,
                theme=st.session_state.theme
            )
            
            st.plotly_chart(fig, use_container_width=True)
            add_download_button(fig, 'comparison_chart', 'Multi_Comparison')

# ============================================================================
# STATS TABLE
# ============================================================================

if len(selected_players) != 2 and selected_players:
    st.markdown("---")
    st.markdown('<div class="section-title">📋 Performance Summary</div>', unsafe_allow_html=True)

    desired_cols = ['player_name', 'matches_played', 'minutes_played'] + pos_info['kpis']
    stats_cols = [c for c in desired_cols if c in clustered_df.columns]
    
    stats_df = clustered_df[clustered_df['player_name'].isin(selected_players)][stats_cols].copy()
    
    rename_dict = {k: config.KPI_READABLE_NAMES.get(k, k) for k in stats_cols}
    rename_dict.update({
        'player_name': 'Player',
        'matches_played': 'Matches',
        'minutes_played': 'Minutes'
    })
    stats_df = stats_df.rename(columns=rename_dict)
    
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

# ============================================================================
# HELP SECTION
# ============================================================================

with st.expander("ℹ️ How to Read the Charts"):
    st.markdown("""
    ### 🍕 Pizza Chart (Single Player)
    - **Slice Length:** Player's **percentile rank** for that attribute
    - **100% (Outer Ring):** League best
    - **50% (Middle):** League average
    - **Colors:** Indicate attribute category

    ### ⚔️ Radar Comparison (Multiple Players)
    - Players are overlaid to show differences
    - Larger area = superior overall performance
    - Click legend items to toggle players
    """)
