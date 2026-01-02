"""
Forex Day Trading AI System
Complete Windows Desktop Application for Multi-Pair Trading with Incremental Learning

Changes from Scalping to Day Trading:
- Target holding period: 6-7 hours (24-28 candles of 15m)
- Risk-Reward Ratio: 1:2
- Uses 15-minute candles for analysis
- Adjusted features for longer timeframes
"""

import os
import sys
import json
import pickle
import logging
import threading
import traceback
from sklearn.base import BaseEstimator, RegressorMixin # Add this line
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import yfinance as yf

# ============================================================================
# CONFIGURATION & CONSTANTS - DAY TRADING VERSION
# ============================================================================

class Config:
    """Application configuration - DAY TRADING EDITION"""
    BASE_DIR = Path("ForexDayTradingAI")
    MODELS_DIR = BASE_DIR / "models"
    SCALERS_DIR = BASE_DIR / "scalers"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"
    
    SEQUENCE_LENGTH = 96  # 24 hours in 15m candles (96 periods)
    FEATURE_LOCK_FILE = "feature_lock.json"
    
    # Training parameters
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    DROPOUT_RATE = 0.3
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # XGBoost parameters
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    
    # DAY TRADING parameters (6-7 hour holds)
    DAYTRADE_TP_PIPS = 40  # Take profit in pips (1:2 RR)
    DAYTRADE_SL_PIPS = 20  # Stop loss in pips
    CONFIDENCE_THRESHOLD = 0.65  # Higher threshold for day trades
    HOLDING_BARS = 28  # 7 hours in 15m bars (28 periods)
    
    # Live data parameters
    LIVE_DAYS = 60  # More data for day trading context
    LIVE_INTERVAL = "15m"
    
    @classmethod
    def create_directories(cls):
        """Create all required directories"""
        for directory in [cls.MODELS_DIR, cls.SCALERS_DIR, cls.LOGS_DIR, cls.DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    """Centralized logging system"""
    
    def __init__(self, log_file: str = "day_trading_system.log"):
        Config.create_directories()
        self.log_file = Config.LOGS_DIR / log_file
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


logger = Logger()


# ============================================================================
# DATA PROCESSING & FEATURE ENGINEERING - DAY TRADING OPTIMIZED
# ============================================================================

class DataProcessor:
    """Handle all data processing, feature engineering, and normalization"""
    
    REQUIRED_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'spread']
    
    def __init__(self):
        self.feature_names = None
        self.scaler = None
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV: {filepath} with shape {df.shape}")
            
            # Validate columns
            missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'spread']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN rows
            df = df.dropna()
            
            logger.info(f"Validated data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {str(e)}")
            raise
    
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
            for period in [20, 50, 100, 200]:  # Longer periods for day trading
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
            df['rsi_28'] = self._calculate_rsi(df['close'], 28)  # Longer for day trading
            
            # MACD with slower settings
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands with 2h window (8 periods of 15m)
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
            
            logger.info(f"Day trading features complete: {df.shape[1]} features")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
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
    
    def create_target(self, df: pd.DataFrame, future_bars: int = None) -> pd.DataFrame:
        """Create target variable for DAY TRADING prediction (6-7 hour horizon)"""
        df = df.copy()
        
        if future_bars is None:
            future_bars = Config.HOLDING_BARS  # 7 hours in 15m bars
        
        # Future price movement for day trading horizon
        df['future_close'] = df['close'].shift(-future_bars)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # DAY TRADING Classification target: 0=SELL, 1=HOLD, 2=BUY
        # Larger thresholds for day trading moves
        df['target'] = 1  # Default HOLD
        df.loc[df['future_return'] > 0.0010, 'target'] = 2  # BUY (100 pips)
        df.loc[df['future_return'] < -0.0010, 'target'] = 0  # SELL
        
        # Regression target: actual return
        df['target_regression'] = df['future_return']
        
        # Remove rows without future data
        df = df.dropna()
        
        logger.info(f"Created day trading target with {future_bars} bars horizon")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        try:
            if sequence_length is None:
                sequence_length = Config.SEQUENCE_LENGTH
            
            # Create target with day trading horizon
            df = self.create_target(df)
            
            # Select feature columns (exclude datetime and target columns)
            exclude_cols = ['datetime', 'target', 'target_regression', 'future_close', 'future_return']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Store feature names for consistency
            self.feature_names = feature_cols
            
            # Extract features and targets
            features = df[feature_cols].values
            target_class = df['target'].values
            target_reg = df['target_regression'].values
            
            # Normalize features
            if self.scaler is None:
                self.scaler = RobustScaler()
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = self.scaler.transform(features)
            
            # Create sequences
            X, y_class, y_reg = [], [], []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y_class.append(target_class[i])
                y_reg.append(target_reg[i])
            
            X = np.array(X)
            y_class = np.array(y_class)
            y_reg = np.array(y_reg)
            
            logger.info(f"Prepared day trading sequences: X={X.shape}, y_class={y_class.shape}, y_reg={y_reg.shape}")
            logger.info(f"Features used: {len(self.feature_names)}")
            
            return X, y_class, y_reg
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def fetch_live_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Fetch live data from yfinance and normalize to training format"""
        try:
            if days is None:
                days = Config.LIVE_DAYS
            
            logger.info(f"Fetching live day trading data for {symbol}...")
            
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
            
            logger.info(f"Fetched {len(df)} rows from yfinance")
            
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
            
            logger.info(f"Normalized live data: {len(df)} rows, {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {str(e)}")
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
                logger.warning(f"Dropping extra features: {extra_features}")
                df = df.drop(columns=list(extra_features))
            
            if missing_features:
                logger.warning(f"Adding missing features with zeros: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0.0
            
            # Reorder columns to match training
            ordered_cols = ['datetime'] + required_features
            df = df[ordered_cols]
            
            logger.info(f"Feature alignment complete: {len(required_features)} features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error aligning features: {str(e)}")
            raise


# ============================================================================
# HYBRID MODEL (TENSORFLOW LSTM + XGBOOST) - DAY TRADING OPTIMIZED
# ============================================================================

class HybridForexModel(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    Hybrid AI model combining LSTM for sequence learning and XGBoost for classification/regression.
    Inherits from BaseEstimator and Mixins to fix the _estimator_type error.
    """
    def __init__(self, model_type='classifier'):
        # Core identification
        self.model_type = model_type
        
        # Model placeholders
        self.lstm_model = None
        self.xgb_model = None
        
        # Explicitly define estimator type for Scikit-learn compatibility
        if model_type == 'regressor':
            self._estimator_type = "regressor"
        else:
            self._estimator_type = "classifier"
            
        # Logging initialization
        logger.info(f"Initialized HybridForexModel as {self._estimator_type}")
        
    def build_lstm_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture optimized for day trading patterns"""
        try:
            model = models.Sequential([
                layers.LSTM(Config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
                layers.Dropout(Config.DROPOUT_RATE),
                layers.LSTM(Config.LSTM_UNITS // 2, return_sequences=True),
                layers.Dropout(Config.DROPOUT_RATE),
                layers.LSTM(Config.LSTM_UNITS // 4, return_sequences=False),
                layers.Dropout(Config.DROPOUT_RATE),
                layers.Dense(Config.DENSE_UNITS, activation='relu'),
                layers.Dropout(Config.DROPOUT_RATE),
                layers.Dense(3, activation='softmax', name='classification_output')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"LSTM model built with input shape {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def train(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray, 
              validation_split: float = 0.2, epochs: int = None, 
              batch_size: int = None, callback_func=None):
        """Train the hybrid model for day trading"""
        try:
            if epochs is None:
                epochs = Config.EPOCHS
            if batch_size is None:
                batch_size = Config.BATCH_SIZE
            
            logger.info(f"Training DAY TRADING model for {self.currency_pair}...")
            logger.info(f"Input shape: {X.shape}, Target shape: {y_class.shape}")
            
            # Store feature count for validation
            self.feature_count = X.shape[2]
            
            # Split data
            X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
                X, y_class, y_reg, test_size=validation_split, random_state=42
            )
            
            # Build or update LSTM model
            if self.lstm_model is None:
                self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
            
            # Custom callback for UI updates
            class UICallback(callbacks.Callback):
                def __init__(self, callback_func):
                    super().__init__()
                    self.callback_func = callback_func
                
                def on_epoch_end(self, epoch, logs=None):
                    if self.callback_func:
                        self.callback_func(epoch, logs)
            
            cb_list = [UICallback(callback_func)] if callback_func else []
            
            # Train LSTM
            history = self.lstm_model.fit(
                X_train, y_class_train,
                validation_data=(X_val, y_class_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=cb_list,
                verbose=1
            )
            
            # Update history
            for key in history.history:
                if key in self.history:
                    self.history[key].extend(history.history[key])
                else:
                    self.history[key] = history.history[key]
            
            # Train XGBoost models
            logger.info("Training XGBoost classifier...")
            
            # Extract LSTM features for XGBoost
            lstm_features_train = self.lstm_model.predict(X_train, verbose=0)
            lstm_features_val = self.lstm_model.predict(X_val, verbose=0)
            
            # XGBoost Classifier
            self.xgb_classifier = xgb.XGBClassifier(
                n_estimators=Config.XGB_N_ESTIMATORS,
                max_depth=Config.XGB_MAX_DEPTH,
                learning_rate=Config.XGB_LEARNING_RATE,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            self.xgb_classifier.fit(lstm_features_train, y_class_train)
            
            # XGBoost Regressor
            logger.info("Training XGBoost regressor...")
            self.xgb_regressor = xgb.XGBRegressor(
                n_estimators=Config.XGB_N_ESTIMATORS,
                max_depth=Config.XGB_MAX_DEPTH,
                learning_rate=Config.XGB_LEARNING_RATE,
                random_state=42
            )
            self.xgb_regressor.fit(lstm_features_train, y_reg_train)
            
            # Validation scores
            val_accuracy = self.xgb_classifier.score(lstm_features_val, y_class_val)
            logger.info(f"XGBoost validation accuracy: {val_accuracy:.4f}")
            
            # Day trading specific validation
            self._validate_day_trading_performance(lstm_features_val, y_class_val)
            
            self.is_trained = True
            logger.info(f"Day trading model training complete for {self.currency_pair}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_day_trading_performance(self, X_val: np.ndarray, y_val: np.ndarray):
        """Additional validation for day trading context"""
        try:
            predictions = self.xgb_classifier.predict(X_val)
            
            # Calculate metrics
            accuracy = np.mean(predictions == y_val)
            
            # Check if model has bias (too many HOLD predictions)
            unique, counts = np.unique(predictions, return_counts=True)
            prediction_dist = dict(zip(unique, counts))
            
            logger.info(f"Day trading validation - Accuracy: {accuracy:.4f}")
            logger.info(f"Prediction distribution: {prediction_dist}")
            
        except Exception as e:
            logger.warning(f"Could not perform day trading validation: {str(e)}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions using hybrid model"""
        try:
            if not self.is_trained:
                raise ValueError("Model has not been trained yet")
            
            # Validate input shape
            if X.shape[2] != self.feature_count:
                raise ValueError(
                    f"Feature dimension mismatch: Expected {self.feature_count} features, "
                    f"got {X.shape[2]} features"
                )
            
            # LSTM predictions
            lstm_probs = self.lstm_model.predict(X, verbose=0)
            
            # XGBoost predictions
            xgb_class = self.xgb_classifier.predict(lstm_probs)
            xgb_probs = self.xgb_classifier.predict_proba(lstm_probs)
            xgb_reg = self.xgb_regressor.predict(lstm_probs)
            
            return xgb_class, xgb_probs, xgb_reg
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save(self, models_dir: Path):
        """Save model components"""
        try:
            pair_dir = models_dir / self.currency_pair
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LSTM model
            lstm_path = pair_dir / "lstm_model.h5"
            self.lstm_model.save(lstm_path)
            
            # Save XGBoost models
            xgb_class_path = pair_dir / "xgb_classifier.json"
            xgb_reg_path = pair_dir / "xgb_regressor.json"
            self.xgb_classifier.save_model(xgb_class_path)
            self.xgb_regressor.save_model(xgb_reg_path)
            
            # Save metadata
            metadata = {
                'currency_pair': self.currency_pair,
                'sequence_length': self.sequence_length,
                'is_trained': self.is_trained,
                'feature_count': self.feature_count,
                'history': self.history,
                'model_type': 'DAY_TRADING',
                'holding_bars': Config.HOLDING_BARS,
                'risk_reward': f"{Config.DAYTRADE_SL_PIPS}:{Config.DAYTRADE_TP_PIPS}"
            }
            metadata_path = pair_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Day trading model saved to {pair_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, models_dir: Path):
        """Load model components"""
        try:
            pair_dir = models_dir / self.currency_pair
            
            if not pair_dir.exists():
                raise FileNotFoundError(f"No saved model found for {self.currency_pair}")
            
            # Load LSTM model
            lstm_path = pair_dir / "lstm_model.h5"
            self.lstm_model = keras.models.load_model(lstm_path)
            
            # Load XGBoost models
            xgb_class_path = pair_dir / "xgb_classifier.json"
            xgb_reg_path = pair_dir / "xgb_regressor.json"
            
            self.xgb_classifier = xgb.XGBClassifier()
            self.xgb_classifier.load_model(xgb_class_path)
            
            self.xgb_regressor = xgb.XGBRegressor()
            self.xgb_regressor.load_model(xgb_reg_path)
            
            # Load metadata
            metadata_path = pair_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.sequence_length = metadata['sequence_length']
            self.is_trained = metadata['is_trained']
            self.feature_count = metadata['feature_count']
            self.history = metadata['history']
            
            logger.info(f"Day trading model loaded from {pair_dir}")
            logger.info(f"Model type: {metadata.get('model_type', 'UNKNOWN')}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


# ============================================================================
# MODEL MANAGER - DAY TRADING VERSION
# ============================================================================

class ModelManager:
    """Manage multiple currency pair models for day trading"""
    
    def __init__(self):
        self.models: Dict[str, HybridForexModel] = {}
        self.processors: Dict[str, DataProcessor] = {}
        self.feature_locks: Dict[str, List[str]] = {}
        
        Config.create_directories()
        self._load_feature_locks()
    
    def _load_feature_locks(self):
        """Load saved feature configurations"""
        lock_file = Config.MODELS_DIR / Config.FEATURE_LOCK_FILE
        if lock_file.exists():
            with open(lock_file, 'r') as f:
                self.feature_locks = json.load(f)
            logger.info(f"Loaded feature locks for {len(self.feature_locks)} pairs")
    
    def _save_feature_locks(self):
        """Save feature configurations"""
        lock_file = Config.MODELS_DIR / Config.FEATURE_LOCK_FILE
        with open(lock_file, 'w') as f:
            json.dump(self.feature_locks, f, indent=2)
        logger.info("Feature locks saved")
    
    def train_model(self, currency_pair: str, csv_path: str, callback_func=None) -> bool:
        """Train or update day trading model for a currency pair"""
        try:
            logger.info(f"Starting DAY TRADING training for {currency_pair}")
            
            # Initialize processor
            if currency_pair not in self.processors:
                self.processors[currency_pair] = DataProcessor()
            
            processor = self.processors[currency_pair]
            
            # Load and process data
            df = processor.load_csv(csv_path)
            df = processor.engineer_features(df)
            
            # Prepare training data with day trading parameters
            X, y_class, y_reg = processor.prepare_training_data(df, Config.SEQUENCE_LENGTH)
            
            # Store feature names
            self.feature_locks[currency_pair] = processor.feature_names
            self._save_feature_locks()
            
            # Save processor (scaler)
            scaler_path = Config.SCALERS_DIR / f"{currency_pair}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(processor.scaler, f)
            
            # Initialize or load model
            if currency_pair not in self.models:
                self.models[currency_pair] = HybridForexModel(currency_pair, Config.SEQUENCE_LENGTH)
            else:
                # Try to load existing model for incremental learning
                try:
                    self.models[currency_pair].load(Config.MODELS_DIR)
                    logger.info(f"Loaded existing day trading model for {currency_pair} - continuing training")
                except:
                    logger.info(f"No existing model found - training from scratch")
            
            # Train model
            model = self.models[currency_pair]
            model.train(X, y_class, y_reg, callback_func=callback_func)
            
            # Save updated model
            model.save(Config.MODELS_DIR)
            
            logger.info(f"Day trading training completed successfully for {currency_pair}")
            return True
            
        except Exception as e:
            logger.error(f"Error training {currency_pair}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_signal(self, currency_pair: str) -> Dict[str, Any]:
        """Generate DAY TRADING signal for currency pair"""
        try:
            logger.info(f"Generating DAY TRADING signal for {currency_pair}")
            
            # Check if model exists
            if currency_pair not in self.models:
                self.models[currency_pair] = HybridForexModel(currency_pair, Config.SEQUENCE_LENGTH)
                self.models[currency_pair].load(Config.MODELS_DIR)
            
            model = self.models[currency_pair]
            
            # Load processor
            if currency_pair not in self.processors:
                self.processors[currency_pair] = DataProcessor()
                scaler_path = Config.SCALERS_DIR / f"{currency_pair}_scaler.pkl"
                with open(scaler_path, 'rb') as f:
                    self.processors[currency_pair].scaler = pickle.load(f)
            
            processor = self.processors[currency_pair]
            
            # Fetch live data
            df_live = processor.fetch_live_data(currency_pair, Config.LIVE_DAYS)
            
            # Engineer features
            df_live = processor.engineer_features(df_live)
            
            # Align features to training
            required_features = self.feature_locks[currency_pair]
            df_live = processor.align_features_for_prediction(df_live, required_features)
            
            # Prepare features for prediction
            feature_cols = [col for col in df_live.columns if col != 'datetime']
            features = df_live[feature_cols].values
            
            # Validate feature count
            if len(feature_cols) != model.feature_count:
                raise ValueError(
                    f"Feature count mismatch: Expected {model.feature_count}, got {len(feature_cols)}"
                )
            
            # Scale features
            features_scaled = processor.scaler.transform(features)
            
            # Create sequence (use last sequence_length candles)
            if len(features_scaled) < Config.SEQUENCE_LENGTH:
                raise ValueError(
                    f"Insufficient data: Need {Config.SEQUENCE_LENGTH} candles, got {len(features_scaled)}"
                )
            
            X = features_scaled[-Config.SEQUENCE_LENGTH:].reshape(1, Config.SEQUENCE_LENGTH, -1)
            
            # Make prediction
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
            
            logger.info(f"Day trading signal generated: {signal_type} with {confidence:.2%} confidence")
            logger.info(f"Risk-Reward: 1:2, Holding: 6-7 hours")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating day trading signal for {currency_pair}: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# ============================================================================
# GUI APPLICATION - DAY TRADING VERSION
# ============================================================================

class ForexDayTradingApp:
    """Main GUI application for DAY TRADING"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Forex Day Trading AI System (6-7 Hour Holds)")
        self.root.geometry("1200x800")
        
        self.model_manager = ModelManager()
        self.training_thread = None
        self.is_training = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Admin Panel (Training)
        self.admin_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.admin_frame, text="Admin / Training")
        self._setup_admin_panel()
        
        # User Panel (Signal Generation)
        self.user_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.user_frame, text="Day Trading Signals")
        self._setup_user_panel()
    
    def _setup_admin_panel(self):
        """Setup admin/training panel"""
        # File upload section
        upload_frame = ttk.LabelFrame(self.admin_frame, text="Upload Training Data", padding=10)
        upload_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(upload_frame, text="Select CSV files to train DAY TRADING models:").pack(anchor='w')
        
        file_frame = ttk.Frame(upload_frame)
        file_frame.pack(fill='x', pady=5)
        
        self.file_listbox = tk.Listbox(file_frame, height=6)
        self.file_listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(file_frame, orient='vertical', command=self.file_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        button_frame = ttk.Frame(upload_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Add CSV Files", command=self._add_files).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear List", command=self._clear_files).pack(side='left', padx=5)
        
        # Training control section
        control_frame = ttk.LabelFrame(self.admin_frame, text="Day Trading Training Control", padding=10)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        info_label = ttk.Label(control_frame, text="Models will be trained for 6-7 hour day trading strategies")
        info_label.pack(anchor='w')
        
        param_label = ttk.Label(control_frame, text=f"Parameters: 1:{Config.DAYTRADE_TP_PIPS/Config.DAYTRADE_SL_PIPS:.0f} RR, {Config.HOLDING_BARS} bars horizon")
        param_label.pack(anchor='w')
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x', pady=10)
        
        self.train_btn = ttk.Button(btn_frame, text="▶ Start Day Trading Training", command=self._start_training)
        self.train_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop Training", command=self._stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.admin_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True)
        
        self.files_to_train = {}
    
    def _setup_user_panel(self):
        """Setup user/signal generation panel for day trading"""
        # Header
        header_frame = ttk.Frame(self.user_frame)
        header_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = ttk.Label(header_frame, text="FOREX DAY TRADING SIGNALS", 
                               font=('Arial', 14, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                  text=f"6-7 Hour Holds • 1:2 Risk-Reward • {Config.LIVE_INTERVAL} Timeframe",
                                  font=('Arial', 10))
        subtitle_label.pack()
        
        # Currency pair selection
        select_frame = ttk.LabelFrame(self.user_frame, text="Currency Pair Selection", padding=10)
        select_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(select_frame, text="Select Currency Pair:").pack(anchor='w')
        
        self.pair_var = tk.StringVar()
        self.pair_combo = ttk.Combobox(select_frame, textvariable=self.pair_var, state='readonly')
        self.pair_combo.pack(fill='x', pady=5)
        
        ttk.Button(select_frame, text="Refresh Pairs", command=self._refresh_pairs).pack(anchor='w', pady=5)
        self._refresh_pairs()
        
        # Analysis control
        analyze_frame = ttk.LabelFrame(self.user_frame, text="Day Trading Analysis", padding=10)
        analyze_frame.pack(fill='x', padx=10, pady=10)
        
        self.analyze_btn = ttk.Button(analyze_frame, text="🔍 Generate Day Trading Signal", 
                                      command=self._generate_signal, style='Accent.TButton')
        self.analyze_btn.pack(pady=10)
        
        # Signal display
        signal_frame = ttk.LabelFrame(self.user_frame, text="Day Trading Signal", padding=10)
        signal_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.signal_text = scrolledtext.ScrolledText(signal_frame, height=20, wrap=tk.WORD, font=('Consolas', 10))
        self.signal_text.pack(fill='both', expand=True)
    
    def _add_files(self):
        """Add CSV files for training"""
        files = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        for filepath in files:
            filename = Path(filepath).stem
            
            # Ask for currency pair name
            pair_name = tk.simpledialog.askstring(
                "Currency Pair",
                f"Enter currency pair name for {Path(filepath).name}:",
                initialvalue=filename.upper()
            )
            
            if pair_name:
                self.files_to_train[pair_name] = filepath
                self.file_listbox.insert(tk.END, f"{pair_name}: {Path(filepath).name}")
    
    def _clear_files(self):
        """Clear file list"""
        self.files_to_train.clear()
        self.file_listbox.delete(0, tk.END)
    
    def _log_message(self, message: str):
        """Log message to training log"""
        self.log_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def _start_training(self):
        """Start training process for day trading"""
        if not self.files_to_train:
            messagebox.showwarning("No Files", "Please add CSV files first!")
            return
        
        self.is_training = True
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_bar.start()
        
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()
    
    def _stop_training(self):
        """Stop training process"""
        self.is_training = False
        self._log_message("Day trading training stopped by user")
    
    def _training_worker(self):
        """Background training worker"""
        try:
            total_pairs = len(self.files_to_train)
            
            for idx, (pair, filepath) in enumerate(self.files_to_train.items(), 1):
                if not self.is_training:
                    break
                
                self._log_message(f"[{idx}/{total_pairs}] Training DAY TRADING model for {pair}...")
                self._log_message(f"  Strategy: 6-7 hour holds, 1:2 Risk-Reward")
                
                def epoch_callback(epoch, logs):
                    if logs:
                        self._log_message(
                            f"  Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, "
                            f"acc={logs.get('accuracy', 0):.4f}, "
                            f"val_loss={logs.get('val_loss', 0):.4f}, "
                            f"val_acc={logs.get('val_accuracy', 0):.4f}"
                        )
                
                success = self.model_manager.train_model(pair, filepath, callback_func=epoch_callback)
                
                if success:
                    self._log_message(f"✓ {pair} day trading model completed successfully")
                else:
                    self._log_message(f"✗ {pair} day trading model training failed")
            
            self._log_message("All day trading training tasks completed!")
            
        except Exception as e:
            self._log_message(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
        
        finally:
            self.is_training = False
            self.root.after(0, self._training_complete)
    
    def _training_complete(self):
        """Cleanup after training"""
        self.progress_bar.stop()
        self.train_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self._refresh_pairs()
        messagebox.showinfo("Training Complete", "Day trading model training has finished!")
    
    def _refresh_pairs(self):
        """Refresh available currency pairs"""
        pairs = []
        
        if Config.MODELS_DIR.exists():
            for item in Config.MODELS_DIR.iterdir():
                if item.is_dir() and (item / "metadata.json").exists():
                    # Check if it's a day trading model
                    try:
                        with open(item / "metadata.json", 'r') as f:
                            metadata = json.load(f)
                        if metadata.get('model_type') == 'DAY_TRADING':
                            pairs.append(item.name)
                    except:
                        pairs.append(item.name)  # Include old models for compatibility
        
        self.pair_combo['values'] = sorted(pairs)
        
        if pairs and not self.pair_var.get():
            self.pair_combo.current(0)
    
    def _generate_signal(self):
        """Generate day trading signal"""
        pair = self.pair_var.get()
        
        if not pair:
            messagebox.showwarning("No Pair Selected", "Please select a currency pair!")
            return
        
        self.analyze_btn.config(state='disabled')
        self.signal_text.delete(1.0, tk.END)
        self.signal_text.insert(tk.END, f"Analyzing {pair} for day trading opportunities...\n\n")
        self.root.update()
        
        def worker():
            try:
                signal = self.model_manager.generate_signal(pair)
                self.root.after(0, lambda: self._display_signal(signal))
            except Exception as e:
                error_msg = f"Error generating day trading signal: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.root.after(0, lambda: self._display_error(error_msg))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _display_signal(self, signal: Dict[str, Any]):
        """Display day trading signal"""
        self.signal_text.delete(1.0, tk.END)
        
        # Format signal output for day trading
        output = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    FOREX DAY TRADING SIGNAL                      ║
║               6-7 Hour Holds • 1:2 Risk-Reward                   ║
╚══════════════════════════════════════════════════════════════════╝

Currency Pair:     {signal['currency_pair']}
Timestamp:         {signal['timestamp']}
Current Price:     {signal['current_price']:.5f}
Model Type:        {signal.get('model_type', 'DAY_TRADING')}
Timeframe:         {signal.get('timeframe', '15m')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DAY TRADING ANALYSIS

Signal:            {signal['signal']}
Confidence:        {signal['confidence']:.2%}
Expected Return:   {signal['expected_return']:.4%}
Holding Time:      {signal.get('holding_time', '6-7 hours')}

Probabilities:
  • SELL:  {signal['probabilities']['SELL']:.2%}
  • HOLD:  {signal['probabilities']['HOLD']:.2%}
  • BUY:   {signal['probabilities']['BUY']:.2%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 DAY TRADING PARAMETERS
"""
        
        if signal['signal'] != 'HOLD':
            output += f"""
Entry Price:       {signal['entry_price']:.5f}
Stop Loss:         {signal['stop_loss']:.5f}
Take Profit:       {signal['take_profit']:.5f}

Risk-Reward Ratio: {signal.get('risk_reward', '1:2')}
Stop Loss Pips:    {Config.DAYTRADE_SL_PIPS}
Take Profit Pips:  {Config.DAYTRADE_TP_PIPS}

Strategy:          6-7 Hour Day Trade
Time Horizon:      {Config.HOLDING_BARS} bars ({Config.HOLDING_BARS * 15} minutes)
"""
        else:
            output += "\n⚠️  No day trade recommended at this time.\n"
            output += "   Market conditions not optimal for 6-7 hour holds.\n"
        
        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ANALYSIS CONTEXT
• Analyzed last {0} candles ({1} minutes)
• Day trading optimized features
• Session-aware analysis (London/NY/Asian)
• 1:2 Risk-Reward strategy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


  DAY TRADING DISCLAIMER:
This signal is for 6-7 hour day trading positions only.
Manage position size appropriately for overnight risk.
Always use stop losses and proper risk management.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".format(signal.get('candles_analyzed', Config.SEQUENCE_LENGTH), signal.get('candles_analyzed', Config.SEQUENCE_LENGTH) * 15)
        
        self.signal_text.insert(tk.END, output)
        self.analyze_btn.config(state='normal')
        
        # Color coding for day trading signals
        if signal['signal'] == 'BUY':
            self.signal_text.tag_add("signal", "12.0", "12.end")
            self.signal_text.tag_config("signal", foreground="green", font=('Consolas', 10, 'bold'))
        elif signal['signal'] == 'SELL':
            self.signal_text.tag_add("signal", "12.0", "12.end")
            self.signal_text.tag_config("signal", foreground="red", font=('Consolas', 10, 'bold'))
        elif signal['signal'] == 'HOLD':
            self.signal_text.tag_add("signal", "12.0", "12.end")
            self.signal_text.tag_config("signal", foreground="orange", font=('Consolas', 10, 'bold'))
    
    def _display_error(self, error_msg: str):
        """Display error message"""
        self.signal_text.delete(1.0, tk.END)
        self.signal_text.insert(tk.END, f"❌ Day Trading Error:\n\n{error_msg}\n\nPlease check the logs for more details.")
        self.analyze_btn.config(state='normal')


# ============================================================================
# MAIN ENTRY POINT - DAY TRADING VERSION
# ============================================================================

def main():
    """Main application entry point for day trading system"""
    try:
        # Create directories
        Config.create_directories()
        
        logger.info("="*80)
        logger.info("FOREX DAY TRADING AI SYSTEM STARTING...")
        logger.info(f"Strategy: 6-7 Hour Holds, 1:2 Risk-Reward")
        logger.info(f"Timeframe: {Config.LIVE_INTERVAL}, Holding Bars: {Config.HOLDING_BARS}")
        logger.info("="*80)
        
        # Initialize Tkinter
        root = tk.Tk()
        
        # Set theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create application
        app = ForexDayTradingApp(root)
        
        # Run main loop
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        messagebox.showerror("Fatal Error", f"Day Trading Application failed to start:\n{str(e)}")


if __name__ == "__main__":

    main()

