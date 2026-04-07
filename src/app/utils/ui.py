import streamlit as st
import config

def display_breadcrumb(page_name="Page"):
    """Display breadcrumb navigation."""
    st.markdown(
        f"""
        <div style='padding: 10px 0; color: #888; font-size: 0.9em;'>
            🏠 <a href='/' target='_self' style='color: #888; text-decoration: none;'>Home</a>
            → 🎯 <b>{page_name}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_position_badge():
    """Display current position filter in sidebar."""
    if 'selected_position' in st.session_state:
        pos_key = st.session_state.selected_position
        # Check if config has POSITIONS, handle if not
        if hasattr(config, 'POSITIONS') and pos_key in config.POSITIONS:
            pos_info = config.POSITIONS[pos_key]
            st.sidebar.markdown(
                f"""
                <div style='background-color: rgba(0,123,255,0.1);
                            padding: 15px;
                            border-radius: 10px;
                            margin-bottom: 20px;'>
                    <h4 style='margin: 0;'>{pos_info.get('icon', '⚽')} {pos_info.get('name', pos_key)}</h4>
                    <p style='margin: 5px 0 0 0; font-size: 0.85em; color: #888;'>
                        {pos_info.get('n_players', 0)} players | {pos_info.get('n_clusters', 0)} clusters
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

def display_loading_skeleton(skeleton_type='chart', height=600):
    """
    Display skeleton loader while data loads.

    Args:
        skeleton_type: Type of skeleton ('chart', 'table', 'scatter', 'heatmap')
        height: Height of skeleton in pixels
    """
    st.markdown(
        f"""
        <div style='background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                    background-size: 200% 100%;
                    animation: loading 1.5s infinite;
                    height: {height}px;
                    border-radius: 10px;
                    margin: 20px 0;'>
        </div>
        <style>
            @keyframes loading {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def display_loading_table_skeleton(rows=5):
    """Display skeleton loader for tables."""
    st.markdown(
        f"""
        <div style='margin: 20px 0;'>
            {''.join([f'''
            <div style='background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                        background-size: 200% 100%;
                        animation: loading 1.5s infinite;
                        height: 40px;
                        border-radius: 5px;
                        margin: 10px 0;'>
            </div>
            ''' for _ in range(rows)])}
        </div>
        <style>
            @keyframes loading {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def display_contextual_help(help_text, icon="ℹ️"):
    """
    Display contextual help tooltip.

    Args:
        help_text: The help text to display
        icon: The icon to use (default: info icon)
    """
    st.markdown(
        f"""
        <span title="{help_text}" style="cursor: help; color: #888; font-size: 1.2em;">
            {icon}
        </span>
        """,
        unsafe_allow_html=True
    )

def inject_presentation_mode_css():
    """Inject CSS for sidebar styling and presentation mode."""
    
    # Always inject sidebar styling
    st.markdown("""
    <style>
    /* ============================================
       SIDEBAR STYLING (Global)
       ============================================ */

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }

    /* Sidebar navigation items */
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        margin-bottom: 0.3rem;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px !important;
        margin: 0 0.5rem !important;
        padding: 0.7rem 1rem !important;
        transition: all 0.3s ease !important;
        border-left: 3px solid transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(240, 147, 251, 0.1) !important;
        border-left-color: #f093fb !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] {
        background: linear-gradient(90deg, rgba(240, 147, 251, 0.2), rgba(245, 87, 108, 0.1)) !important;
        border-left-color: #f093fb !important;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        color: #e0e0e0 !important;
    }

    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 1rem 0 !important;
    }

    /* Presentation mode checkbox */
    [data-testid="stSidebar"] .stCheckbox {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 10px !important;
        padding: 0.8rem !important;
        margin: 0 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox label {
        color: #e0e0e0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Presentation mode specific styling
    if st.session_state.get('presentation_mode', False):
        st.markdown(
            """
            <style>
            /* Hide sidebar in presentation mode */
            [data-testid="stSidebar"] {
                display: none !important;
            }

            /* Larger fonts for better visibility */
            .main .block-container {
                padding-top: 2rem;
                max-width: 95% !important;
            }

            h1 { font-size: 3rem !important; }
            h2 { font-size: 2.5rem !important; }
            h3 { font-size: 2rem !important; }
            p, div, span { font-size: 1.3rem !important; }

            /* Hide expanders and captions */
            .streamlit-expanderHeader { display: none; }
            [data-testid="stExpander"] { display: none; }
            .stCaption { display: none; }

            /* Make metrics larger */
            [data-testid="stMetric"] {
                font-size: 1.5rem !important;
            }

            /* Larger buttons */
            .stButton button {
                font-size: 1.5rem !important;
                padding: 1rem 2rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
