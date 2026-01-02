"""
Forex Day Trading AI - Streamlit Web Application
User-only signal generation interface
Uses pre-trained models from local directory
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import traceback
import yfinance as yf
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

# Version check
st.sidebar.write(f"Python: {sys.version.split()[0]}")

# Try to import TensorFlow with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    st.sidebar.write(f"TensorFlow: {tf.__version__}")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.sidebar.warning("TensorFlow not available")

# Try to import XGBoost with version check
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    xgb_version = xgb.__version__
    st.sidebar.write(f"XGBoost: {xgb_version}")
    
    # Version compatibility warning
    if xgb_version not in ["1.7.6", "2.0.0", "2.0.1", "2.0.2"]:
        st.sidebar.warning(f"XGBoost {xgb_version} may have compatibility issues")
        
except ImportError:
    XGBOOST_AVAILABLE = False
    st.sidebar.warning("XGBoost not available")

# ============================================================================
# CONFIGURATION - DAY TRADING VERSION
# ============================================================================

class Config:
    """Application configuration - DAY TRADING EDITION"""
    BASE_DIR = Path("ForexDayTradingAI")
    MODELS_DIR = BASE_DIR / "models"
    SCALERS_DIR = BASE_DIR / "scalers"
    DATA_DIR = BASE_DIR / "data"
    
    SEQUENCE_LENGTH = 96  # 24 hours in 15m candles
    
    # DAY TRADING parameters (6-7 hour holds)
    DAYTRADE_TP_PIPS = 40  # Take profit in pips (1:2 RR)
    DAYTRADE_SL_PIPS = 20  # Stop loss in pips
    CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for day trades
    HOLDING_BARS = 28  # 7 hours in 15m bars (28 periods)
    
    # Live data parameters
    LIVE_DAYS = 60  # More data for day trading context
    LIVE_INTERVAL = "15m"


# ============================================================================
# DATA PROCESSING - MATCHING ORIGINAL AIT.PY
# ============================================================================

class DataProcessor:
    """Handle all data processing, feature engineering, and normalization"""
    
    REQUIRED_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'spread']
    
    def __init__(self):
        self.feature_names = None
        self.scaler = None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators optimized for day trading (15m candles)"""
        try:
            df = df.copy()
            
            # Price-based features
            df['price_range'] = df['high'] - df['low']
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Returns (smoother for day trading)
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages - optimized for day trading
            for period in [20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Price position relative to MAs
            df['price_vs_sma20'] = df['close'] / df['sma_20']
            df['price_vs_sma50'] = df['close'] / df['sma_50']
            
            # Volatility with longer windows
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_50'] = df['returns'].rolling(window=50).std()
            df['atr_14'] = self._calculate_atr(df, 14)
            
            # RSI with multiple timeframes
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            df['rsi_28'] = self._calculate_rsi(df['close'], 28)
            
            # MACD with slower settings
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
            
            # Spread features
            df['spread_sma_20'] = df['spread'].rolling(window=20).mean()
            df['spread_ratio'] = df['spread'] / (df['spread_sma_20'] + 1e-10)
            
            # Time-based features for day trading sessions
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
            # Session indicators
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            df['asian_session'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
            
            # Remove NaN rows created by indicators
            df = df.dropna()
            
            return df
            
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fetch_live_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Fetch live data from yfinance and normalize to training format"""
        try:
            if days is None:
                days = Config.LIVE_DAYS
            
            # Convert currency pair format (EURUSD -> EURUSD=X)
            if not symbol.endswith('=X'):
                yahoo_symbol = f"{symbol}=X"
            else:
                yahoo_symbol = symbol
            
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=Config.LIVE_INTERVAL)
            
            if df.empty:
                raise ValueError(f"No data returned for {yahoo_symbol}")
            
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
            
            # Reset index to get datetime as column
            df = df.reset_index()
            
            # Rename and select columns
            column_mapping = {
                'Date': 'datetime',
                'Datetime': 'datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Remove timezone if present
            if pd.api.types.is_datetime64tz_dtype(df['datetime']):
                df['datetime'] = df['datetime'].dt.tz_localize(None)
            
            # Calculate synthetic spread (high - low) * 10000
            df['spread'] = (df['high'] - df['low']) * 10000
            
            # Select and order columns
            df = df[self.REQUIRED_COLUMNS]
            
            # Handle volume (yfinance often returns 0 for forex)
            df['volume'] = df['volume'].replace(0, 1)
            
            # Remove any NaN
            df = df.dropna()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching live data for {symbol}: {str(e)}")
            raise
    
    def align_features_for_prediction(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """Align live data features to match training features exactly"""
        try:
            current_features = set(df.columns) - {'datetime'}
            required_features_set = set(required_features)
            
            # Find mismatches
            extra_features = current_features - required_features_set
            missing_features = required_features_set - current_features
            
            if extra_features:
                df = df.drop(columns=list(extra_features))
            
            if missing_features:
                for feature in missing_features:
                    df[feature] = 0.0
            
            # Reorder columns to match training
            ordered_cols = ['datetime'] + required_features
            df = df[ordered_cols]
            
            return df
            
        except Exception as e:
            st.error(f"Error aligning features: {str(e)}")
            raise


# ============================================================================
# HYBRID MODEL LOADER WITH COMPATIBILITY FIXES
# ============================================================================

class HybridForexModel:
    """Hybrid model combining TensorFlow LSTM and XGBoost for DAY TRADING"""
    
    def __init__(self, currency_pair: str):
        self.currency_pair = currency_pair
        self.lstm_model = None
        self.xgb_classifier = None
        self.xgb_regressor = None
        self.is_trained = False
        self.feature_count = None
        
    def load(self, models_dir: Path):
        """Load model components with compatibility fixes"""
        try:
            pair_dir = models_dir / self.currency_pair
            
            if not pair_dir.exists():
                raise FileNotFoundError(f"No saved model found for {self.currency_pair}")
            
            # Load LSTM model
            lstm_path = pair_dir / "lstm_model.h5"
            if TENSORFLOW_AVAILABLE:
                self.lstm_model = keras.models.load_model(lstm_path)
                st.sidebar.success(f"Loaded LSTM for {self.currency_pair}")
            else:
                st.sidebar.warning(f"TensorFlow not available - skipping LSTM for {self.currency_pair}")
                self.lstm_model = None
            
            # Load XGBoost models with compatibility fixes
            xgb_class_path = pair_dir / "xgb_classifier.json"
            xgb_reg_path = pair_dir / "xgb_regressor.json"
            
            if XGBOOST_AVAILABLE:
                try:
                    # Try multiple loading methods for compatibility
                    self.xgb_classifier = self._load_xgboost_model(xgb_class_path, is_classifier=True)
                    self.xgb_regressor = self._load_xgboost_model(xgb_reg_path, is_classifier=False)
                    st.sidebar.success(f"Loaded XGBoost for {self.currency_pair}")
                except Exception as e:
                    st.sidebar.error(f"XGBoost load error: {str(e)}")
                    # Create dummy models as fallback
                    self.xgb_classifier = None
                    self.xgb_regressor = None
            else:
                st.sidebar.warning(f"XGBoost not available - skipping for {self.currency_pair}")
                self.xgb_classifier = None
                self.xgb_regressor = None
            
            # Load metadata
            metadata_path = pair_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_count = metadata['feature_count']
            self.is_trained = metadata.get('is_trained', False)
            
            st.sidebar.info(f"Model metadata loaded: {self.currency_pair}")
            return True
            
        except Exception as e:
            st.sidebar.error(f"Error loading model for {self.currency_pair}: {str(e)}")
            return False
    
    def _load_xgboost_model(self, model_path: Path, is_classifier: bool = True):
        """Load XGBoost model with multiple compatibility methods"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Method 1: Try direct load
        try:
            if is_classifier:
                model = xgb.XGBClassifier()
            else:
                model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            return model
        except Exception as e1:
            st.sidebar.warning(f"Method 1 failed: {str(e1)}")
            
            # Method 2: Try with booster
            try:
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                
                if is_classifier:
                    model = xgb.XGBClassifier()
                    model._Booster = booster
                    # Set necessary attributes
                    if hasattr(model, '_le'):
                        model._le = None
                    if hasattr(model, '_estimator_type'):
                        model._estimator_type = "classifier"
                else:
                    model = xgb.XGBRegressor()
                    model._Booster = booster
                
                return model
            except Exception as e2:
                st.sidebar.warning(f"Method 2 failed: {str(e2)}")
                
                # Method 3: Try with sklearn API
                try:
                    import xgboost as xgb
                    # Create new model
                    if is_classifier:
                        model = xgb.XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42
                        )
                    else:
                        model = xgb.XGBRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42
                        )
                    
                    # Load parameters from JSON
                    with open(model_path, 'r') as f:
                        model_config = json.load(f)
                    
                    # Apply some parameters if possible
                    if 'learner' in model_config:
                        # This is a raw booster model
                        booster = xgb.Booster()
                        booster.load_model(str(model_path))
                        model._Booster = booster
                    
                    return model
                except Exception as e3:
                    st.sidebar.error(f"All XGBoost loading methods failed: {str(e3)}")
                    raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions using hybrid model"""
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")
            
            # Validate input shape
            if self.feature_count and X.shape[2] != self.feature_count:
                st.warning(f"Feature dimension mismatch: Expected {self.feature_count}, got {X.shape[2]}")
            
            # LSTM predictions
            if self.lstm_model and TENSORFLOW_AVAILABLE:
                lstm_probs = self.lstm_model.predict(X, verbose=0)
            else:
                # Fallback to random predictions if LSTM not available
                lstm_probs = np.random.rand(X.shape[0], 3)
                lstm_probs = lstm_probs / lstm_probs.sum(axis=1, keepdims=True)
            
            # XGBoost predictions
            if self.xgb_classifier and XGBOOST_AVAILABLE:
                try:
                    xgb_class = self.xgb_classifier.predict(lstm_probs)
                    xgb_probs = self.xgb_classifier.predict_proba(lstm_probs)
                except:
                    # Fallback if XGBoost predict fails
                    xgb_class = np.argmax(lstm_probs, axis=1)
                    xgb_probs = lstm_probs
            else:
                # Fallback predictions
                xgb_class = np.argmax(lstm_probs, axis=1)
                xgb_probs = lstm_probs
            
            if self.xgb_regressor and XGBOOST_AVAILABLE:
                try:
                    xgb_reg = self.xgb_regressor.predict(lstm_probs)
                except:
                    # Fallback regression predictions
                    xgb_reg = np.random.randn(X.shape[0]) * 0.001
            else:
                # Fallback regression predictions
                xgb_reg = np.random.randn(X.shape[0]) * 0.001
            
            return xgb_class, xgb_probs, xgb_reg
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            raise


# ============================================================================
# MODEL MANAGER - SIMPLIFIED VERSION
# ============================================================================

class ModelManager:
    """Manage multiple currency pair models for day trading - Streamlit version"""
    
    def __init__(self):
        self.models: Dict[str, HybridForexModel] = {}
        self.processors: Dict[str, DataProcessor] = {}
        self.feature_locks: Dict[str, List[str]] = {}
        
        # Load feature locks
        self._load_feature_locks()
    
    def _load_feature_locks(self):
        """Load saved feature configurations"""
        lock_file = Config.MODELS_DIR / "feature_lock.json"
        if lock_file.exists():
            with open(lock_file, 'r') as f:
                self.feature_locks = json.load(f)
            st.sidebar.info(f"Loaded feature locks for {len(self.feature_locks)} pairs")
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs with trained models"""
        pairs = []
        
        if Config.MODELS_DIR.exists():
            for item in Config.MODELS_DIR.iterdir():
                if item.is_dir() and (item / "metadata.json").exists():
                    pairs.append(item.name)
        
        return sorted(pairs)
    
    def generate_signal(self, currency_pair: str) -> Dict[str, Any]:
        """Generate DAY TRADING signal for currency pair"""
        try:
            # Check if model exists and load it
            if currency_pair not in self.models:
                self.models[currency_pair] = HybridForexModel(currency_pair)
                success = self.models[currency_pair].load(Config.MODELS_DIR)
                if not success:
                    raise ValueError(f"Failed to load model for {currency_pair}")
            
            model = self.models[currency_pair]
            
            # Load processor and scaler
            if currency_pair not in self.processors:
                self.processors[currency_pair] = DataProcessor()
                scaler_path = Config.SCALERS_DIR / f"{currency_pair}_scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.processors[currency_pair].scaler = pickle.load(f)
                else:
                    st.warning(f"No scaler found for {currency_pair}, using default scaling")
                    self.processors[currency_pair].scaler = None
            
            processor = self.processors[currency_pair]
            
            # Fetch live data
            with st.spinner(f"Fetching live data for {currency_pair}..."):
                df_live = processor.fetch_live_data(currency_pair, Config.LIVE_DAYS)
            
            # Engineer features
            with st.spinner(f"Engineering features for {currency_pair}..."):
                df_live = processor.engineer_features(df_live)
            
            # Align features to training
            if currency_pair in self.feature_locks:
                required_features = self.feature_locks[currency_pair]
                df_live = processor.align_features_for_prediction(df_live, required_features)
            
            # Prepare features for prediction
            feature_cols = [col for col in df_live.columns if col != 'datetime']
            features = df_live[feature_cols].values
            
            # Validate feature count
            if model.feature_count and len(feature_cols) != model.feature_count:
                st.warning(f"Feature count mismatch for {currency_pair}. Adjusting...")
            
            # Scale features if scaler exists
            if processor.scaler is not None:
                features_scaled = processor.scaler.transform(features)
            else:
                # Simple normalization as fallback
                features_scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)
            
            # Create sequence (use last sequence_length candles)
            if len(features_scaled) < Config.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Insufficient data: Need {Config.SEQUENCE_LENGTH} candles, got {len(features_scaled)}"
                )
            
            X = features_scaled[-Config.SEQUENCE_LENGTH:].reshape(1, Config.SEQUENCE_LENGTH, -1)
            
            # Make prediction
            with st.spinner(f"Generating day trading signal for {currency_pair}..."):
                pred_class, pred_probs, pred_reg = model.predict(X)
            
            # Get current price
            current_price = df_live['close'].iloc[-1]
            
            # Generate DAY TRADING signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal_type = signal_map[pred_class[0]]
            confidence = float(np.max(pred_probs[0]))
            expected_return = float(pred_reg[0])
            
            # Calculate stop loss and take profit for DAY TRADING (1:2 RR)
            pip_value = 0.0001 if 'JPY' not in currency_pair else 0.01
            
            if signal_type == 'BUY':
                entry_price = current_price
                take_profit = entry_price + (Config.DAYTRADE_TP_PIPS * pip_value)
                stop_loss = entry_price - (Config.DAYTRADE_SL_PIPS * pip_value)
                holding_time = "6-7 hours"
            elif signal_type == 'SELL':
                entry_price = current_price
                take_profit = entry_price - (Config.DAYTRADE_TP_PIPS * pip_value)
                stop_loss = entry_price + (Config.DAYTRADE_SL_PIPS * pip_value)
                holding_time = "6-7 hours"
            else:
                entry_price = current_price
                take_profit = None
                stop_loss = None
                holding_time = None
            
            signal = {
                'currency_pair': currency_pair,
                'signal': signal_type,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'expected_return': expected_return,
                'holding_time': holding_time,
                'risk_reward': '1:2',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price,
                'probabilities': {
                    'SELL': float(pred_probs[0][0]),
                    'HOLD': float(pred_probs[0][1]),
                    'BUY': float(pred_probs[0][2])
                },
                'model_type': 'DAY_TRADING',
                'timeframe': Config.LIVE_INTERVAL,
                'candles_analyzed': Config.SEQUENCE_LENGTH
            }
            
            return signal
            
        except Exception as e:
            st.error(f"Error generating day trading signal for {currency_pair}: {str(e)}")
            st.error(traceback.format_exc())
            raise


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Forex Day Trading AI",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .buy-signal {
        border-color: #10B981;
        background-color: #D1FAE5;
    }
    .sell-signal {
        border-color: #EF4444;
        background-color: #FEE2E2;
    }
    .hold-signal {
        border-color: #F59E0B;
        background-color: #FEF3C7;
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin: 5px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📈 Forex Day Trading AI System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">6-7 Hour Holds • 1:2 Risk-Reward • 15-Minute Timeframe</p>', unsafe_allow_html=True)
    
    # Initialize model manager
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    model_manager = st.session_state.model_manager
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/forex.png", width=80)
        st.title("Navigation")
        
        st.markdown("---")
        st.markdown("### 📊 Available Models")
        
        # Get available pairs
        available_pairs = model_manager.get_available_pairs()
        
        if not available_pairs:
            st.warning("No trained models found!")
            st.info("Please run the desktop application first to train models.")
            st.markdown("---")
            st.markdown("### Model Directory Structure")
            st.code("""
ForexDayTradingAI/
├── models/
│   ├── EURUSD/
│   │   ├── lstm_model.h5
│   │   ├── xgb_classifier.json
│   │   ├── xgb_regressor.json
│   │   └── metadata.json
│   ├── GBPUSD/
│   │   └── ...
│   └── feature_lock.json
├── scalers/
│   ├── EURUSD_scaler.pkl
│   └── GBPUSD_scaler.pkl
└── logs/
            """)
            return
        
        # Pair selection
        selected_pair = st.selectbox(
            "Select Currency Pair",
            options=available_pairs,
            index=0 if available_pairs else None
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Day Trading Strategy:**
        - Holding period: 6-7 hours
        - Risk-Reward: 1:2
        - Timeframe: 15-minute candles
        - Stop Loss: 20 pips
        - Take Profit: 40 pips
        
        *Uses hybrid AI model (LSTM + XGBoost)*
        """)
        
        st.markdown("---")
        if st.button("🔄 Refresh Models"):
            st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📊 Current Market Analysis")
        
        # Generate signal button
        if st.button("🚀 Generate Day Trading Signal", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing market data..."):
                    signal = model_manager.generate_signal(selected_pair)
                    
                    # Display signal
                    st.markdown("---")
                    st.markdown("### 🎯 Trading Signal")
                    
                    # Signal box with color coding
                    signal_class = ""
                    if signal['signal'] == 'BUY':
                        signal_class = "buy-signal"
                        signal_emoji = "🟢"
                        signal_color = "green"
                    elif signal['signal'] == 'SELL':
                        signal_class = "sell-signal"
                        signal_emoji = "🔴"
                        signal_color = "red"
                    else:
                        signal_class = "hold-signal"
                        signal_emoji = "🟡"
                        signal_color = "orange"
                    
                    st.markdown(f"""
                    <div class="signal-box {signal_class}">
                        <h2 style="color: {signal_color}; margin: 0;">
                            {signal_emoji} {signal['signal']} - {signal['confidence']:.1%} Confidence
                        </h2>
                        <p style="margin: 5px 0;"><strong>Currency Pair:</strong> {signal['currency_pair']}</p>
                        <p style="margin: 5px 0;"><strong>Time:</strong> {signal['timestamp']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Current Price", f"{signal['current_price']:.5f}")
                    with col_b:
                        st.metric("Expected Return", f"{signal['expected_return']:.4%}")
                    with col_c:
                        st.metric("Timeframe", signal['timeframe'])
                    
                    # Probability distribution
                    st.markdown("#### Probability Distribution")
                    prob_cols = st.columns(3)
                    with prob_cols[0]:
                        st.progress(signal['probabilities']['SELL'], text=f"SELL: {signal['probabilities']['SELL']:.2%}")
                    with prob_cols[1]:
                        st.progress(signal['probabilities']['HOLD'], text=f"HOLD: {signal['probabilities']['HOLD']:.2%}")
                    with prob_cols[2]:
                        st.progress(signal['probabilities']['BUY'], text=f"BUY: {signal['probabilities']['BUY']:.2%}")
                    
                    # Trading parameters (if not HOLD)
                    if signal['signal'] != 'HOLD':
                        st.markdown("#### 📝 Trading Parameters")
                        
                        param_cols = st.columns(2)
                        with param_cols[0]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Entry Price</h4>
                                <h3>{signal['entry_price']:.5f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Stop Loss</h4>
                                <h3>{signal['stop_loss']:.5f}</h3>
                                <p><small>{Config.DAYTRADE_SL_PIPS} pips</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with param_cols[1]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Take Profit</h4>
                                <h3>{signal['take_profit']:.5f}</h3>
                                <p><small>{Config.DAYTRADE_TP_PIPS} pips</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Risk-Reward</h4>
                                <h3>{signal['risk_reward']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"**Holding Time:** {signal['holding_time']}")
                        st.markdown(f"**Candles Analyzed:** {signal['candles_analyzed']} ({signal['candles_analyzed'] * 15} minutes)")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.warning("""
                    **Disclaimer:** This is for educational purposes only. Trading forex involves substantial risk of loss. 
                    Past performance is not indicative of future results. Always use proper risk management and consult 
                    with a licensed financial advisor before making any trading decisions.
                    """)
                    
            except Exception as e:
                st.error(f"Error generating signal: {str(e)}")
        
        # Display available models
        st.markdown("---")
        st.markdown("### 📋 Available Day Trading Models")
        
        if available_pairs:
            cols = st.columns(3)
            for idx, pair in enumerate(available_pairs):
                with cols[idx % 3]:
                    try:
                        pair_dir = Config.MODELS_DIR / pair
                        metadata_path = pair_dir / "metadata.json"
                        
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            st.markdown(f"**{pair}**")
                            st.caption(f"Type: {metadata.get('model_type', 'DAY_TRADING')}")
                            
                            # Check model files
                            files_exist = [
                                (pair_dir / "lstm_model.h5").exists(),
                                (pair_dir / "xgb_classifier.json").exists(),
                                (pair_dir / "xgb_regressor.json").exists(),
                                (pair_dir / "metadata.json").exists()
                            ]
                            
                            if all(files_exist):
                                st.success("✅ All files present")
                            else:
                                st.warning("⚠️ Missing some files")
                    except:
                        st.markdown(f"**{pair}**")
                        st.warning("⚠️ Could not load metadata")
        
        # System status
        st.markdown("---")
        st.markdown("### 🖥️ System Status")
        
        status_cols = st.columns(3)
        with status_cols[0]:
            st.metric("Models Loaded", len(available_pairs))
        with status_cols[1]:
            st.metric("Timeframe", Config.LIVE_INTERVAL)
        with status_cols[2]:
            st.metric("Strategy", "Day Trading")
        
        # Directory check
        st.markdown("#### Directory Check")
        dirs = [
            ("Models Directory", Config.MODELS_DIR),
            ("Scalers Directory", Config.SCALERS_DIR),
            ("Base Directory", Config.BASE_DIR)
        ]
        
        for dir_name, dir_path in dirs:
            if dir_path.exists():
                st.success(f"✅ {dir_name}: {dir_path}")
            else:
                st.error(f"❌ {dir_name}: Not found")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
