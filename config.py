"""
Configuration settings for OilX Supply & Demand Forecasting App
"""

# Application settings
APP_TITLE = "OilX Supply & Demand Forecaster"
APP_ICON = "🛢️"
APP_VERSION = "1.0.0"

# Data settings
DATA_FILE = "SupplyDemand_TopOilCountries_Under24MB.csv"
CACHE_TTL = 3600  # Cache timeout in seconds (1 hour)

# UI settings
THEME_PRIMARY_COLOR = "#1E3A8A"  # Dark blue
THEME_SECONDARY_COLOR = "#4F46E5"  # Indigo
THEME_ACCENT_COLOR = "#EF4444"  # Red
THEME_SUCCESS_COLOR = "#059669"  # Green
THEME_WARNING_COLOR = "#DC2626"  # Bright red
THEME_BACKGROUND_COLOR = "#f0f2f6"  # Light gray

# Page names
PAGES = {
    "DASHBOARD": "Dashboard",
    "EXPLORER": "Supply & Demand Explorer",
    "FORECASTING": "Forecasting",
    "MODEL_INSIGHTS": "Model Insights",
    "SCENARIO_ANALYSIS": "Scenario Analysis",
    "CORRELATION_ANALYSIS": "Correlation Analysis"
}

# Default selections
DEFAULT_COUNTRIES = ["United States", "China", "Russia", "Saudi Arabia", "India"]
DEFAULT_PRODUCT = "Crude Oil"
DEFAULT_FLOW = "Production"

# Forecasting model configurations
FORECAST_MODELS = {
    "SARIMA": {
        "name": "SARIMA",
        "description": "Statistical time series model with seasonal components",
        "params": {
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 12)
        }
    },
    "PROPHET": {
        "name": "Prophet",
        "description": "Facebook's forecasting model for time series",
        "params": {
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "growth": "linear",
            "seasonality_mode": "additive"
        }
    },
    "XGBOOST": {
        "name": "XGBoost",
        "description": "Gradient boosting machine learning model",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "reg:squarederror"
        }
    },
    "LSTM": {
        "name": "LSTM",
        "description": "Deep learning model for complex patterns",
        "params": {
            "units": 64,
            "dropout": 0.2,
            "epochs": 50
        }
    },
    "ENSEMBLE": {
        "name": "Ensemble",
        "description": "Combination of multiple models",
        "params": {
            "models": ["SARIMA", "PROPHET", "XGBOOST"]
        }
    }
}

# Feature engineering configurations
FEATURE_ENGINEERING_CONFIG = {
    "TIME_FEATURES": {
        "enabled": True,
        "description": "Calendar-based features like month, year, etc."
    },
    "LAG_FEATURES": {
        "enabled": True,
        "description": "Lagged values of the target variable",
        "max_lag": 12
    },
    "ROLLING_FEATURES": {
        "enabled": True,
        "description": "Rolling window statistics",
        "windows": [3, 6, 12],
        "functions": ["mean", "std", "min", "max"]
    },
    "DIFF_FEATURES": {
        "enabled": True,
        "description": "Differenced values of the target variable",
        "periods": [1, 12]
    },
    "SEASONAL_FEATURES": {
        "enabled": True,
        "description": "Seasonal components from decomposition",
        "period": 12
    },
    "FOURIER_FEATURES": {
        "enabled": True,
        "description": "Fourier terms for capturing seasonality",
        "period": 12,
        "terms": 3
    }
}

# Time aggregation options
TIME_AGGREGATIONS = {
    "MONTHLY": {
        "name": "Monthly",
        "freq": "MS",
        "min_periods": 24,
        "display_format": "%b %Y"
    },
    "QUARTERLY": {
        "name": "Quarterly",
        "freq": "QS",
        "min_periods": 8,
        "display_format": "Q%q %Y"
    },
    "YEARLY": {
        "name": "Yearly",
        "freq": "YS",
        "min_periods": 3,
        "display_format": "%Y"
    }
}

# Country groupings
COUNTRY_GROUPS = {
    "MAJOR_PRODUCERS": [
        "United States", "Saudi Arabia", "Russia", 
        "Canada", "China", "Iraq", "United Arab Emirates",
        "Brazil", "Kuwait", "Mexico"
    ],
    "MAJOR_CONSUMERS": [
        "United States", "China", "India", "Japan",
        "South Korea", "Germany", "Russia", "Brazil",
        "Saudi Arabia", "Canada"
    ],
    "OPEC": [
        "Algeria", "Angola", "Congo", "Equatorial Guinea",
        "Gabon", "Iran", "Iraq", "Kuwait", "Libya", "Nigeria",
        "Saudi Arabia", "United Arab Emirates", "Venezuela"
    ],
    "EUROPE": [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "Estonia", "Finland", "France",
        "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia",
        "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
        "Portugal", "Romania", "Slovakia", "Slovenia", "Spain",
        "Sweden", "United Kingdom"
    ],
    "ASIA_PACIFIC": [
        "China", "Japan", "India", "South Korea", "Indonesia",
        "Australia", "Thailand", "Taiwan", "Malaysia", "Singapore",
        "Philippines", "Vietnam", "New Zealand", "Bangladesh"
    ]
}

# Product groupings
PRODUCT_GROUPS = {
    "CRUDE_AND_CONDENSATE": [
        "Crude Oil", "Condensate"
    ],
    "REFINED_PRODUCTS": [
        "Gasoline", "Diesel", "Jet Fuel", "Kerosene", 
        "Fuel Oil", "LPG", "Naphtha"
    ],
    "NATURAL_GAS": [
        "Natural Gas", "LNG"
    ],
    "PETROCHEMICALS": [
        "Ethylene", "Propylene", "Benzene", "Xylene", "Toluene"
    ]
}

# Flow groupings
FLOW_GROUPS = {
    "SUPPLY": [
        "Production", "Imports", "Refinery Output"
    ],
    "DEMAND": [
        "Consumption", "Exports", "Refinery Input"
    ],
    "INVENTORY": [
        "Stocks", "Stock Change"
    ]
}

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning {
        color: #ff4b4b;
        font-weight: bold;
    }
    .success {
        color: #00c853;
        font-weight: bold;
    }
    .info-card {
        background-color: #f8fafc;
        border-left: 4px solid #4F46E5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px 4px 0px 0px;
    }
</style>
"""