"""
Streamlit Web UI for Forex Day Trading AI System
Simplified blocking version - no threading issues
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

# ============================================================================
# IMPORT EXISTING CLASSES WITHOUT TKINTER CONFLICTS
# ============================================================================

import unittest.mock as mock

# Create mock for tkinter modules
mock_tkinter = mock.MagicMock()
mock_tkinter.Tk = mock.MagicMock()
mock_tkinter.ttk = mock.MagicMock()
mock_tkinter.messagebox = mock.MagicMock()
mock_tkinter.filedialog = mock.MagicMock()
mock_tkinter.scrolledtext = mock.MagicMock()

# Patch tkinter before importing ait
sys.modules['tkinter'] = mock_tkinter
sys.modules['tkinter.ttk'] = mock_tkinter.ttk
sys.modules['tkinter.messagebox'] = mock_tkinter.messagebox
sys.modules['tkinter.filedialog'] = mock_tkinter.filedialog
sys.modules['tkinter.scrolledtext'] = mock_tkinter.scrolledtext

try:
    # Import the original module
    import ait
    
    # Get the classes we need
    Config = ait.Config
    logger = ait.logger
    
    # Import ModelManager directly
    from ait import ModelManager
    
    # Create a global model manager instance
    model_manager = ModelManager()
    
    st.success("✅ Forex Day Trading AI System Loaded")
    
except Exception as e:
    st.error(f"Error importing AI system: {str(e)}")
    st.stop()

# ============================================================================
# STREAMLIT UI - SIMPLE BLOCKING VERSION
# ============================================================================

st.set_page_config(
    page_title="Forex Day Trading AI",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font { font-size: 24px; font-weight: bold; }
    .signal-box { 
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        margin: 10px 0;
    }
    .buy { color: green; font-weight: bold; }
    .sell { color: red; font-weight: bold; }
    .hold { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Forex Day Trading AI System")
st.markdown("**6-7 Hour Holds • 1:2 Risk-Reward • 15-Minute Timeframe**")

# Function to get available pairs
def get_available_pairs():
    pairs = []
    if Config.MODELS_DIR.exists():
        for item in Config.MODELS_DIR.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                try:
                    with open(item / "metadata.json", 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('model_type') == 'DAY_TRADING':
                        pairs.append(item.name)
                except:
                    pairs.append(item.name)
    return sorted(pairs)

# Get available pairs
available_pairs = get_available_pairs()

# Create tabs
tab1, tab2 = st.tabs(["📈 Generate Signals", "⚙️ Train Models"])

with tab1:
    st.header("Generate Day Trading Signals")
    
    if not available_pairs:
        st.warning("⚠️ No trained models found. Please train models first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_pair = st.selectbox(
                "Select Currency Pair:",
                available_pairs,
                key="signal_pair"
            )
            
            if st.button("🚀 Generate Signal", type="primary", use_container_width=True):
                # Store in session state to trigger signal generation
                st.session_state.generate_signal = True
                st.session_state.selected_pair = selected_pair
        
        with col2:
            # Check if we need to generate a signal
            if 'generate_signal' in st.session_state and st.session_state.generate_signal:
                selected_pair = st.session_state.selected_pair
                
                # Generate signal (blocking - will show spinner)
                with st.spinner(f"Analyzing {selected_pair}..."):
                    try:
                        # Generate the signal
                        signal = model_manager.generate_signal(selected_pair)
                        
                        # Display the signal
                        st.markdown("---")
                        st.markdown(f"### 📊 {selected_pair} Day Trading Signal")
                        
                        # Signal type
                        if signal['signal'] == 'BUY':
                            st.markdown('<div class="buy">🟢 BUY SIGNAL</div>', unsafe_allow_html=True)
                        elif signal['signal'] == 'SELL':
                            st.markdown('<div class="sell">🔴 SELL SIGNAL</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="hold">🟡 HOLD SIGNAL</div>', unsafe_allow_html=True)
                        
                        # Key metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Confidence", f"{signal['confidence']:.2%}")
                            st.metric("Current Price", f"{signal['current_price']:.5f}")
                        
                        with col_b:
                            if signal['signal'] != 'HOLD':
                                st.metric("Entry Price", f"{signal['entry_price']:.5f}")
                                st.metric("Stop Loss", f"{signal['stop_loss']:.5f}")
                            else:
                                st.metric("Strategy", "No Trade")
                        
                        with col_c:
                            if signal['signal'] != 'HOLD':
                                st.metric("Take Profit", f"{signal['take_profit']:.5f}")
                                st.metric("Expected Return", f"{signal['expected_return']:.4%}")
                        
                        # Probabilities
                        st.markdown("#### Signal Probabilities")
                        prob_cols = st.columns(3)
                        with prob_cols[0]:
                            st.progress(signal['probabilities']['SELL'])
                            st.metric("SELL", f"{signal['probabilities']['SELL']:.2%}")
                        with prob_cols[1]:
                            st.progress(signal['probabilities']['HOLD'])
                            st.metric("HOLD", f"{signal['probabilities']['HOLD']:.2%}")
                        with prob_cols[2]:
                            st.progress(signal['probabilities']['BUY'])
                            st.metric("BUY", f"{signal['probabilities']['BUY']:.2%}")
                        
                        # Trading parameters
                        st.markdown("#### Trading Parameters")
                        st.info(f"**Timeframe**: {signal.get('timeframe', '15m')} | "
                               f"**Holding**: {signal.get('holding_time', '6-7 hours')} | "
                               f"**Risk-Reward**: {signal.get('risk_reward', '1:2')}")
                        
                        # Additional info
                        with st.expander("View Full Signal Details"):
                            st.json(signal)
                        
                        st.caption(f"Generated: {signal['timestamp']}")
                        
                        # Clear the flag
                        del st.session_state.generate_signal
                        
                    except Exception as e:
                        st.error(f"Error generating signal: {str(e)}")
                        del st.session_state.generate_signal

with tab2:
    st.header("Train New Models")
    
    st.info("""
    **Training Process:**
    1. Upload CSV files with historical data
    2. Assign currency pair names
    3. Train day trading models (6-7 hour holds)
    """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload CSV files for training",
        type=['csv'],
        accept_multiple_files=True,
        help="Each file should contain historical Forex data for one currency pair"
    )
    
    if uploaded_files:
        st.write(f"📁 **{len(uploaded_files)} file(s) uploaded**")
        
        # Collect files for training
        training_files = {}
        
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            default_pair = Path(filename).stem.upper().replace('_', '').replace('-', '')
            
            col1, col2 = st.columns([3, 1])
            with col1:
                pair_name = st.text_input(
                    f"Pair name for {filename}",
                    value=default_pair,
                    key=f"pair_{filename}"
                )
            
            with col2:
                if st.button("Add", key=f"add_{filename}"):
                    # Save file
                    temp_path = Config.DATA_DIR / filename
                    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    training_files[pair_name] = str(temp_path)
                    st.success(f"Added {pair_name}")
        
        # Training button
        if training_files:
            st.markdown("---")
            st.write("**Files ready for training:**")
            for pair, path in training_files.items():
                st.write(f"• {pair}: {Path(path).name}")
            
            if st.button("🎯 Start Training", type="primary"):
                with st.spinner("Training models..."):
                    # Simple training without callbacks
                    for pair, filepath in training_files.items():
                        try:
                            st.write(f"Training {pair}...")
                            success = model_manager.train_model(pair, filepath)
                            if success:
                                st.success(f"✓ {pair} trained successfully")
                            else:
                                st.error(f"✗ {pair} training failed")
                        except Exception as e:
                            st.error(f"Error training {pair}: {str(e)}")
                    
                    st.success("✅ All models trained!")
                    # Refresh available pairs
                    available_pairs = get_available_pairs()
                    st.rerun()

# Sidebar
with st.sidebar:
    st.header("System Info")
    
    # Create directories
    Config.create_directories()
    
    st.metric("Trained Pairs", len(available_pairs))
    st.metric("Models Path", str(Config.MODELS_DIR))
    
    st.markdown("---")
    st.markdown("**Available Pairs:**")
    if available_pairs:
        for pair in available_pairs:
            st.write(f"• {pair}")
    else:
        st.write("None")
    
    # Refresh button
    if st.button("🔄 Refresh Pairs"):
        available_pairs = get_available_pairs()
        st.rerun()
    
    st.markdown("---")
    st.caption("Forex Day Trading AI v1.0")

# Footer
st.markdown("---")
st.caption("Day Trading Strategy: 6-7 hour holds • 1:2 Risk-Reward • 15-minute timeframe")
