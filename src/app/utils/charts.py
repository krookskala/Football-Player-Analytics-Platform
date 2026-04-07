import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import config

def create_pizza_chart(
    player_name: str,
    kpis: List[str],
    percentiles: pd.Series,
    raw_values: pd.Series,
    theme: str = 'light'
) -> go.Figure:
    """
    Create a 'Pizza Chart' (Percentile Radar) for a single player.
    
    Args:
        player_name: Name of the player
        kpis: List of KPI names to display
        percentiles: Series containing percentile scores (0-100) for KPIs
        raw_values: Series containing raw values for KPIs (for hover)
        theme: 'light' or 'dark'
        
    Returns:
        Plotly Figure
    """
    theme_config = config.THEMES[theme]
    
    # Prepare data
    values = []
    hover_texts = []
    colors = []
    categories = []
    readable_kpis = []
    
    for kpi in kpis:
        # Get readable name
        readable_name = config.KPI_READABLE_NAMES.get(kpi, kpi)
        readable_kpis.append(readable_name)
        
        # Get percentile (length of slice)
        val = percentiles.get(kpi, 0)
        values.append(val)
        
        # Get category and color
        cat = config.KPI_CATEGORIES.get(kpi, 'Other')
        categories.append(cat)
        colors.append(config.CATEGORY_COLORS.get(cat, '#9E9E9E'))
        
        # Hover text
        raw = raw_values.get(kpi, 0)
        hover_texts.append(f"<b>{readable_name}</b><br>Percentile: {val:.0f}<br>Value: {raw:.2f}")

    # Create figure
    fig = go.Figure()
    
    # Add bars (pizza slices)
    fig.add_trace(go.Barpolar(
        r=values,
        theta=readable_kpis,
        text=values,
        marker_color=colors,
        marker_line_color=theme_config['background'],
        marker_line_width=2,
        opacity=0.8,
        hoverinfo='text',
        hovertext=hover_texts,
        name=player_name
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b><br><span style='font-size:12px'>Percentile Rank vs Position</span>",
            y=0.95
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10, color='gray'),
                gridcolor=theme_config.get('grid_color', '#E0E0E0')
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=theme_config['text']),
                gridcolor=theme_config.get('grid_color', '#E0E0E0')
            ),
            bgcolor=theme_config['background']
        ),
        paper_bgcolor=theme_config['background'],
        plot_bgcolor=theme_config['background'],
        font=dict(color='black' if theme == 'light' else 'white'),
        showlegend=False,
        margin=dict(t=80, b=50, l=100, r=100), # Increased margins
        height=500
    )
    
    return fig

def create_comparison_chart(
    players_data: List[Dict],
    kpis: List[str],
    theme: str = 'light'
) -> go.Figure:
    """
    Create a comparison chart (Grouped Bar or Radar) for multiple players.
    For 2 players, we use a 'Butterfly' style bar chart or Overlaid Radar.
    Here we implement an Overlaid Radar with filled areas, as it's standard.
    
    Args:
        players_data: List of dicts, each having {'name': str, 'percentiles': Series, 'raw': Series}
        kpis: List of KPIs
        theme: 'light' or 'dark'
    """
    theme_config = config.THEMES[theme]
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080'] # Pure Red, Blue, Green, Orange, Purple for maximum contrast
    
    # Get readable names
    readable_kpis = [config.KPI_READABLE_NAMES.get(k, k) for k in kpis]
    
    fig = go.Figure()
    
    for idx, p_data in enumerate(players_data):
        name = p_data['name']
        p_scores = p_data['percentiles']
        
        # Get values
        values = [p_scores.get(k, 0) for k in kpis]
        
        # Close the loop
        values_closed = values + [values[0]]
        kpis_closed = readable_kpis + [readable_kpis[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=kpis_closed,
            fill='toself',
            name=name,
            line_color=colors[idx % len(colors)],
            opacity=0.6
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor=theme_config['background']
        ),
        paper_bgcolor=theme_config['background'],
        plot_bgcolor=theme_config['background'],
        font=dict(color='black' if theme == 'light' else 'white'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600, # Increased height
        margin=dict(t=100, b=50, l=120, r=120) # Increased margins significantly
    )
    
    return fig


import plotly.io as pio
import streamlit as st

def add_download_button(fig, filename_prefix, key):
    """Add centered download buttons with hover effects for Plotly figure."""
    
    # Add custom CSS for button styling
    st.markdown("""
    <style>
    .stDownloadButton > button {
        background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        padding: 0.4rem 1rem !important;
        color: #fff !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
        white-space: nowrap !important;
        min-width: 80px !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(145deg, rgba(240, 147, 251, 0.25) 0%, rgba(245, 87, 108, 0.15) 100%) !important;
        border-color: #f093fb !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.25) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered columns with fixed widths
    spacer1, col1, col2, spacer2 = st.columns([3, 1, 1, 3])

    with col1:
        try:
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            st.download_button(
                label="📥 PNG",
                data=img_bytes,
                file_name=f"{filename_prefix}.png",
                mime="image/png",
                key=f"{key}_png"
            )
        except Exception:
            st.caption("PNG N/A")

    with col2:
        html_bytes = pio.to_html(fig, include_plotlyjs='cdn').encode()
        st.download_button(
            label="📥 HTML",
            data=html_bytes,
            file_name=f"{filename_prefix}.html",
            mime="text/html",
            key=f"{key}_html"
        )
    
    return spacer1, col1, col2, spacer2
