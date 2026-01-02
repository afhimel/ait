"""
Streamlit Web UI for Forex Day Trading AI System
Replaces Tkinter GUI while keeping all backend logic identical
"""

import streamlit as st
import threading
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Import existing classes from the original project
# Note: We'll import only what we need to avoid Tkinter conflicts
try:
    # Import Config and logger directly
    from ait import Config, logger
    
    # We need to import ModelManager but avoid Tkinter dependencies
    # We'll create a modified import approach
    import sys
    import os
    
    # Temporarily disable tkinter imports in the module
    original_import = __builtins__.__import__
    
    def custom_import(name, *args, **kwargs):
        if name in ['tkinter', 'tk', 'Tkinter']:
            raise ImportError(f"Tkinter disabled for Streamlit")
        return original_import(name, *args, **kwargs)
    
    __builtins__.__import__ = custom_import
    
    # Now import ModelManager
    from ait import ModelManager
    
    # Restore original import
    __builtins__.__import__ = original_import
    
except Exception as e:
    st.error(f"Error importing from original project: {str(e)}")
    st.stop()

# Initialize session state variables
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []

if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

if 'training_files' not in st.session_state:
    st.session_state.training_files = {}

if 'available_pairs' not in st.session_state:
    st.session_state.available_pairs = []

if 'current_signal' not in st.session_state:
    st.session_state.current_signal = None

if 'signal_generating' not in st.session_state:
    st.session_state.signal_generating = False


# Function to update available pairs
def update_available_pairs():
    """Refresh list of available trained currency pairs"""
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
    
    st.session_state.available_pairs = sorted(pairs)
    return pairs


# Training thread worker function
def training_worker(files_to_train: Dict[str, str]):
    """Background worker for training models"""
    try:
        total_pairs = len(files_to_train)
        
        for idx, (pair, filepath) in enumerate(files_to_train.items(), 1):
            if not st.session_state.training_in_progress:
                break
            
            log_message(f"[{idx}/{total_pairs}] Training DAY TRADING model for {pair}...")
            log_message(f"  Strategy: 6-7 hour holds, 1:2 Risk-Reward")
            
            def epoch_callback(epoch, logs):
                if logs:
                    log_message(
                        f"  Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, "
                        f"acc={logs.get('accuracy', 0):.4f}, "
                        f"val_loss={logs.get('val_loss', 0):.4f}, "
                        f"val_acc={logs.get('val_accuracy', 0):.4f}"
                    )
            
            # Train the model using existing logic
            success = st.session_state.model_manager.train_model(
                pair, 
                filepath, 
                callback_func=epoch_callback
            )
            
            if success:
                log_message(f"✓ {pair} day trading model completed successfully")
            else:
                log_message(f"✗ {pair} day trading model training failed")
        
        if st.session_state.training_in_progress:
            log_message("All day trading training tasks completed!")
        
    except Exception as e:
        log_message(f"Training error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        st.session_state.training_in_progress = False
        update_available_pairs()


# Signal generation worker function
def signal_worker(currency_pair: str):
    """Background worker for generating signals"""
    try:
        signal = st.session_state.model_manager.generate_signal(currency_pair)
        st.session_state.current_signal = signal
    except Exception as e:
        st.session_state.current_signal = {
            'error': f"Error generating signal: {str(e)}",
            'currency_pair': currency_pair,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        import traceback
        logger.error(traceback.format_exc())
    finally:
        st.session_state.signal_generating = False


# Helper function for logging
def log_message(message: str):
    """Add message to training logs"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted_message = f"{timestamp} - {message}"
    st.session_state.training_logs.append(formatted_message)
    
    # Keep only last 1000 lines to prevent memory issues
    if len(st.session_state.training_logs) > 1000:
        st.session_state.training_logs = st.session_state.training_logs[-1000:]


# Streamlit UI Setup
st.set_page_config(
    page_title="Forex Day Trading AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-card {
        background-color: #F9FAFB;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .training-log {
        background-color: #1F2937;
        color: #D1D5DB;
        font-family: monospace;
        padding: 1rem;
        border-radius: 5px;
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .buy-signal { color: #10B981; font-weight: bold; }
    .sell-signal { color: #EF4444; font-weight: bold; }
    .hold-signal { color: #F59E0B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Forex Day Trading AI System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">6-7 Hour Holds • 1:2 Risk-Reward • 15-Minute Timeframe</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📊 Day Trading Signals", "⚙️ Admin / Training"])

with tab1:
    st.header("Day Trading Signal Generation")
    
    # Update available pairs
    update_available_pairs()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Select Currency Pair")
        
        if not st.session_state.available_pairs:
            st.warning("No trained models found. Please train models in the Admin tab first.")
            selected_pair = None
        else:
            selected_pair = st.selectbox(
                "Choose a currency pair:",
                st.session_state.available_pairs,
                index=0 if st.session_state.available_pairs else None
            )
        
        generate_btn = st.button(
            "🔍 Generate Day Trading Signal",
            type="primary",
            disabled=st.session_state.signal_generating or not selected_pair,
            use_container_width=True
        )
        
        if generate_btn and selected_pair:
            st.session_state.signal_generating = True
            st.session_state.current_signal = None
            
            # Start signal generation in background thread
            thread = threading.Thread(
                target=signal_worker,
                args=(selected_pair,),
                daemon=True
            )
            thread.start()
            
            with st.spinner(f"Analyzing {selected_pair} for day trading opportunities..."):
                while st.session_state.signal_generating:
                    time.sleep(0.1)
    
    with col2:
        st.subheader("Signal Analysis")
        
        if st.session_state.current_signal:
            signal = st.session_state.current_signal
            
            if 'error' in signal:
                st.error(f"Error: {signal['error']}")
            else:
                # Display signal in a nice format
                st.markdown(f"### {signal['currency_pair']} Day Trading Signal")
                
                # Signal type with color coding
                if signal['signal'] == 'BUY':
                    st.markdown(f'<h2 class="buy-signal">📈 BUY Signal</h2>', unsafe_allow_html=True)
                elif signal['signal'] == 'SELL':
                    st.markdown(f'<h2 class="sell-signal">📉 SELL Signal</h2>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<h2 class="hold-signal">⏸️ HOLD Signal</h2>', unsafe_allow_html=True)
                
                # Create columns for metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Confidence", f"{signal['confidence']:.2%}")
                    st.metric("Current Price", f"{signal['current_price']:.5f}")
                
                with col_b:
                    if signal['signal'] != 'HOLD':
                        st.metric("Entry Price", f"{signal['entry_price']:.5f}")
                        st.metric("Stop Loss", f"{signal['stop_loss']:.5f}" if signal['stop_loss'] else "N/A")
                
                with col_c:
                    if signal['signal'] != 'HOLD':
                        st.metric("Take Profit", f"{signal['take_profit']:.5f}" if signal['take_profit'] else "N/A")
                        st.metric("Expected Return", f"{signal['expected_return']:.4%}")
                
                # Probabilities
                st.subheader("Signal Probabilities")
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
                st.subheader("Day Trading Parameters")
                
                param_cols = st.columns(2)
                with param_cols[0]:
                    st.info(f"**Timeframe**: {signal.get('timeframe', '15m')}")
                    st.info(f"**Holding Time**: {signal.get('holding_time', '6-7 hours')}")
                    st.info(f"**Risk-Reward**: {signal.get('risk_reward', '1:2')}")
                
                with param_cols[1]:
                    st.info(f"**Stop Loss Pips**: {Config.DAYTRADE_SL_PIPS}")
                    st.info(f"**Take Profit Pips**: {Config.DAYTRADE_TP_PIPS}")
                    st.info(f"**Model Type**: {signal.get('model_type', 'DAY_TRADING')}")
                
                # Timestamp
                st.caption(f"Generated at: {signal['timestamp']}")
                
                # Raw signal data (expandable)
                with st.expander("View Raw Signal Data"):
                    st.json(signal)
        else:
            st.info("Select a currency pair and click 'Generate Day Trading Signal' to begin analysis.")

with tab2:
    st.header("Admin Panel - Model Training")
    
    # Training file upload section
    st.subheader("Upload Training Data")
    
    uploaded_files = st.file_uploader(
        "Select CSV files for training",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload CSV files with historical Forex data"
    )
    
    # Display uploaded files and collect pair names
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Collect pair names for each file
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.training_files:
                # Default pair name from filename
                default_pair = Path(uploaded_file.name).stem.upper()
                
                # Get user input for pair name
                col1, col2 = st.columns([3, 1])
                with col1:
                    pair_name = st.text_input(
                        f"Currency pair name for {uploaded_file.name}",
                        value=default_pair,
                        key=f"pair_{uploaded_file.name}"
                    )
                with col2:
                    if st.button("Add", key=f"add_{uploaded_file.name}"):
                        # Save file temporarily
                        temp_path = Config.DATA_DIR / uploaded_file.name
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.session_state.training_files[pair_name] = str(temp_path)
                        st.success(f"Added {pair_name}")
                        st.rerun()
        
        # Show current training files
        if st.session_state.training_files:
            st.subheader("Files Ready for Training")
            for pair, path in st.session_state.training_files.items():
                st.write(f"**{pair}**: {Path(path).name}")
            
            # Clear all button
            if st.button("Clear All Files"):
                st.session_state.training_files = {}
                st.rerun()
    
    # Training control section
    st.subheader("Training Control")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Day Trading Strategy**: 6-7 hour holds, 1:{Config.DAYTRADE_TP_PIPS/Config.DAYTRADE_SL_PIPS:.0f} Risk-Reward")
        st.info(f"**Training Parameters**: {Config.EPOCHS} epochs, {Config.BATCH_SIZE} batch size")
    
    with col2:
        start_disabled = st.session_state.training_in_progress or not st.session_state.training_files
        stop_disabled = not st.session_state.training_in_progress
        
        if st.button(
            "▶ Start Day Trading Training",
            disabled=start_disabled,
            type="primary",
            use_container_width=True
        ):
            st.session_state.training_in_progress = True
            st.session_state.training_logs = []  # Clear previous logs
            
            # Start training in background thread
            thread = threading.Thread(
                target=training_worker,
                args=(st.session_state.training_files,),
                daemon=True
            )
            thread.start()
            
            st.rerun()
        
        if st.button(
            "⏹ Stop Training",
            disabled=stop_disabled,
            type="secondary",
            use_container_width=True
        ):
            st.session_state.training_in_progress = False
            log_message("Training stopped by user")
            st.rerun()
    
    # Training progress and logs
    st.subheader("Training Progress")
    
    if st.session_state.training_in_progress:
        st.warning("⚠️ Training in progress... Please do not close this browser tab.")
        progress_bar = st.progress(0)
        
        # Simple progress indicator (in real app, you'd want more sophisticated tracking)
        import random
        progress_bar.progress(random.randint(10, 90) / 100)
    
    # Training logs display
    st.subheader("Training Logs")
    
    if st.session_state.training_logs:
        # Create a scrollable log display
        log_container = st.container()
        with log_container:
            # Reverse to show newest first
            for log_line in reversed(st.session_state.training_logs[-50:]):  # Show last 50 lines
                st.code(log_line, language=None)
    else:
        st.info("Training logs will appear here once training starts.")
    
    # Clear logs button
    if st.button("Clear Logs"):
        st.session_state.training_logs = []
        st.rerun()

# Sidebar with system info
with st.sidebar:
    st.header("System Information")
    
    st.metric("Models Directory", str(Config.MODELS_DIR))
    st.metric("Data Directory", str(Config.DATA_DIR))
    
    # Show available trained models
    st.subheader("Trained Models")
    if st.session_state.available_pairs:
        for pair in st.session_state.available_pairs[:10]:  # Show first 10
            st.text(f"✓ {pair}")
        if len(st.session_state.available_pairs) > 10:
            st.caption(f"... and {len(st.session_state.available_pairs) - 10} more")
    else:
        st.text("No trained models")
    
    # Refresh button
    if st.button("🔄 Refresh Models List"):
        update_available_pairs()
        st.rerun()
    
    # System status
    st.subheader("System Status")
    
    if st.session_state.training_in_progress:
        st.error("⚠️ Training Active")
    elif st.session_state.signal_generating:
        st.warning("🔍 Generating Signal")
    else:
        st.success("✅ System Ready")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    if st.button("View Training Directory"):
        import subprocess
        import platform
        
        path = Config.MODELS_DIR
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer "{path}"')
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

# Auto-refresh when training is in progress
if st.session_state.training_in_progress or st.session_state.signal_generating:
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.caption("Forex Day Trading AI System v1.0 • 6-7 Hour Holds • 1:2 Risk-Reward")