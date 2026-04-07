"""
Demo Modes for Thesis Defense
-------------------------------
Predefined scenarios that load instantly with one click.

Usage:
    from demo_modes import display_demo_buttons, load_demo_scenario

    # In Home page
    display_demo_buttons()

    # Or load programmatically
    load_demo_scenario('forward_comparison_case_study')
"""

import streamlit as st
import config

# Page mapping with professional academic filenames
PAGE_MAPPING = {
    'Scouting_Tool': 'pages/Scouting_Tool.py',
    'K_Means_Clustering_Analysis': 'pages/K_Means_Clustering_Analysis.py',
    'Random_Forest_Validation': 'pages/Random_Forest_Validation.py',
    'Dashboard': 'pages/Dashboard.py'
}

# ============================================================================
# DEMO SCENARIO LOADER
# ============================================================================

def load_demo_scenario(scenario_name: str):
    """
    Load predefined demo scenario into session_state.

    Args:
        scenario_name: One of 'forward_comparison_case_study', 'clustering_visualization_case_study', 'model_validation_case_study'

    Triggers:
        - Updates session_state with scenario parameters
        - Switches to appropriate page
        - Reruns Streamlit

    Example:
        load_demo_scenario('forward_comparison_case_study')
        # → Loads Messi vs Giroud comparison on Player Comparison page
    """
    if scenario_name not in config.DEMO_SCENARIOS:
        st.error(f"❌ Unknown demo scenario: {scenario_name}")
        return

    scenario = config.DEMO_SCENARIOS[scenario_name]

    # Set demo mode flag
    st.session_state.demo_mode = True
    st.session_state.demo_scenario = scenario_name

    # Update session state based on scenario
    st.session_state.selected_position = scenario.get('position')

    if scenario_name == 'forward_comparison_case_study':
        # Player comparison scenario
        st.session_state.selected_players = scenario['players']
        st.session_state.selected_kpis = scenario['kpis']
        st.session_state.active_page = 'Scouting_Tool'

        # Success message
        st.success(
            f"✅ Demo Mode Loaded: **{scenario['name']}**\n\n"
            f"📍 Position: {config.POSITIONS[scenario['position']]['name']}\n\n"
            f"👥 Players: {', '.join(scenario['players'])}"
        )

    elif scenario_name == 'clustering_visualization_case_study':
        # Cluster PCA scenario
        st.session_state.selected_viz = scenario['view']
        st.session_state.highlight_cluster = scenario.get('highlight_cluster')
        st.session_state.active_page = 'K_Means_Clustering_Analysis'

        st.success(
            f"✅ Demo Mode Loaded: **{scenario['name']}**\n\n"
            f"📍 Position: {config.POSITIONS[scenario['position']]['name']}\n\n"
            f"📊 View: PCA Scatter (Cluster {scenario.get('highlight_cluster')} highlighted)"
        )

    elif scenario_name == 'model_validation_case_study':
        # RF validation scenario
        st.session_state.selected_view = scenario['view']
        st.session_state.active_page = 'Random_Forest_Validation'

        st.success(
            f"✅ Demo Mode Loaded: **{scenario['name']}**\n\n"
            f"📍 Position: {config.POSITIONS[scenario['position']]['name']}\n\n"
            f"📊 View: RF vs F-Statistics Correlation"
        )

    # Give user time to read the message before switching pages
    import time
    time.sleep(1.5)

    # Switch to appropriate page
    target_page = scenario['page']
    
    target_file = PAGE_MAPPING.get(target_page)

    if not target_file:
        st.error(f"❌ Page configuration error: {target_page}")
        return

    # Streamlit page switching
    try:
        st.switch_page(target_file)
    except Exception as e:
        st.warning(
            f"⚠️ Page transition failed ({e}). Please select the page from the left menu."
        )


# ============================================================================
# DEMO BUTTONS DISPLAY
# ============================================================================

def display_demo_buttons():
    """
    Display demo mode buttons on home page.

    Shows:
        - 3 predefined demo scenarios as clickable buttons
        - Descriptions for each scenario
        - One-click loading

    Usage:
        In pages/1_Home.py:
            from demo_modes import display_demo_buttons
            display_demo_buttons()
    """
    st.subheader("🎬 Thesis Defense Demo Modes")

    st.markdown(
        """
        **Quick Demo:** Load pre-configured scenarios with one click.
        Start demos instantly with 1 button instead of 5 dropdowns for thesis defense.
        """
    )

    # Create 3 columns for 3 demo buttons
    col1, col2, col3 = st.columns(3)

    # Case Study 1: Forward Comparison
    with col1:
        with st.container():
            st.markdown("### ⚽ Case Study 1")
            st.caption(config.DEMO_SCENARIOS['forward_comparison_case_study']['description'])
            st.markdown(
                f"**Position:** {config.POSITIONS['forward']['name']}\n\n"
                f"**Players:** 2 forwards\n\n"
                f"**View:** Radar Chart"
            )

            if st.button(
                "🚀 Run Case Study 1",
                use_container_width=True,
                type="primary",
                key="demo_case_1"
            ):
                load_demo_scenario('forward_comparison_case_study')

    # Case Study 2: Clustering Visualization
    with col2:
        with st.container():
            st.markdown("### 📊 Case Study 2")
            st.caption(config.DEMO_SCENARIOS['clustering_visualization_case_study']['description'])
            st.markdown(
                f"**Position:** {config.POSITIONS['midfielder']['name']}\n\n"
                f"**Clusters:** 2 clusters\n\n"
                f"**View:** PCA Scatter"
            )

            if st.button(
                "🚀 Run Case Study 2",
                use_container_width=True,
                type="primary",
                key="demo_case_2"
            ):
                load_demo_scenario('clustering_visualization_case_study')

    # Case Study 3: Model Validation
    with col3:
        with st.container():
            st.markdown("### 🔍 Case Study 3")
            st.caption(config.DEMO_SCENARIOS['model_validation_case_study']['description'])
            st.markdown(
                f"**Position:** {config.POSITIONS['midfielder']['name']}\n\n"
                f"**Metric:** RF Accuracy\n\n"
                f"**View:** Correlation Scatter"
            )

            if st.button(
                "🚀 Run Case Study 3",
                use_container_width=True,
                type="primary",
                key="demo_case_3"
            ):
                load_demo_scenario('model_validation_case_study')

    # Instructions
    st.divider()

    with st.expander("💡 How to Use Demo Modes?"):
        st.markdown(
            """
            **What are demo modes?**
            - Pre-configured scenarios for quick use during thesis defense
            - All parameters loaded automatically with 1 button

            **How to use?**
            1. Click one of the 3 buttons above
            2. Page will automatically switch to the relevant demo
            3. Charts will load instantly

            **When to use?**
            - ✅ For quick demonstrations to thesis committee
            - ✅ To test the dashboard
            - ✅ To see example scenarios

            **Want manual control?**
            - Select desired page from left menu
            - Set your own parameters
            """
        )


# ============================================================================
# DEMO MODE INDICATOR
# ============================================================================

def display_demo_mode_indicator():
    """
    Display indicator if currently in demo mode.

    Shows:
        - Badge indicating demo mode is active
        - Current scenario name
        - Button to exit demo mode

    Usage:
        In sidebar of any page:
            from demo_modes import display_demo_mode_indicator
            display_demo_mode_indicator()
    """
    if st.session_state.get('demo_mode', False):
        scenario_name = st.session_state.get('demo_scenario', 'Unknown')
        scenario = config.DEMO_SCENARIOS.get(scenario_name, {})

        st.sidebar.info(
            f"🎬 **DEMO MODE ACTIVE**\n\n"
            f"Scenario: **{scenario.get('name', 'Unknown')}**"
        )

        if st.sidebar.button("❌ Exit Demo Mode", use_container_width=True):
            exit_demo_mode()


def exit_demo_mode():
    """
    Exit demo mode and reset session state.
    """
    st.session_state.demo_mode = False
    st.session_state.demo_scenario = None

    # Reset selections (optional)
    if st.sidebar.checkbox("Reset selections too?", value=True):
        st.session_state.selected_players = []
        st.session_state.selected_position = config.UI_SETTINGS['default_position']

    st.success("✅ Exited demo mode")
    st.rerun()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_active_demo_scenario():
    """
    Get currently active demo scenario.

    Returns:
        Dict with scenario config or None
    """
    if not st.session_state.get('demo_mode', False):
        return None

    scenario_name = st.session_state.get('demo_scenario')
    return config.DEMO_SCENARIOS.get(scenario_name)


def is_demo_mode_active():
    """
    Check if demo mode is currently active.

    Returns:
        bool
    """
    return st.session_state.get('demo_mode', False)


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_demo_state():
    """
    Initialize demo-related session state.

    Call this in app.py on startup.
    """
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False

    if 'demo_scenario' not in st.session_state:
        st.session_state.demo_scenario = None

    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'Dashboard'


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("🧪 Testing demo_modes.py...")

    # Test scenario loading
    for scenario_name in config.DEMO_SCENARIOS.keys():
        scenario = config.DEMO_SCENARIOS[scenario_name]
        print(f"\n📍 Scenario: {scenario['name']}")
        print(f"   Position: {scenario.get('position')}")
        print(f"   Page: {scenario.get('page')}")

        if 'players' in scenario:
            print(f"   Players: {scenario['players']}")
        if 'view' in scenario:
            print(f"   View: {scenario['view']}")

    print("\n✅ Demo modes module tests passed!")
