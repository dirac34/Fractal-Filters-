# Financial Time Series Forecasting with Fractal Analysis

A comprehensive framework integrating fractal analysis and deep learning architectures for enhanced financial volatility prediction across multiple market regimes.

## Overview

This project implements advanced volatility forecasting models that combine traditional econometric methods (ARIMA, GARCH) with modern deep learning architectures (LSTM, GRU, CNN, CapsNet) enhanced by fractal feature extraction techniques.

### Key Features

- **Multi-Regime Analysis**: Bull, Bear, and Recovery market periods (2012-2025)
- **Fractal Enhancement**: Hurst exponent and wavelet-based feature extraction
- **Comprehensive Modeling**: Statistical and deep learning approaches
- **Production-Ready**: Optimized implementations with checkpoint systems
- **Statistical Validation**: Diebold-Mariano testing for model comparison

## Project Structure

```
├── comprehensive_analysis.py    # Full feature analysis (24-48 hours)
├── optimized_analysis.py       # Fast analysis (8-12 hours)  
├── data/                       # Market data files
├── results/                    # Output files and checkpoints
├── plots/                      # Visualization outputs
└── README.md                   # This file
```

## Requirements

### Core Dependencies

```python
# Data Science & Analysis
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0

# Financial & Statistical Analysis
statsmodels>=0.13.0
arch>=5.3.0                    # GARCH models
scipy>=1.7.0

# Signal Processing & Fractals
PyWavelets>=1.1.1              # Wavelet transforms
nolds>=0.5.2                   # Fractal analysis (Hurst exponent)

# Deep Learning
tensorflow>=2.8.0
keras>=2.8.0

# Utilities
tqdm>=4.62.0                   # Progress bars
pickle                         # Checkpoint system
warnings
gc                            # Memory management
```

### Installation

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
pip install statsmodels arch scipy
pip install PyWavelets nolds
pip install tensorflow

# Or install all at once
pip install -r requirements.txt
```

## Quick Start

### 1. Comprehensive Analysis (24-48 hours)

```python
# Full analysis with all features and models
python comprehensive_analysis.py
```

**Features:**
- 4 market indicators: DIX, GEX, SKEW, PUTCALLRATIO
- 9 model types: ARIMA, GARCH, ANN, RNN, LSTM, GRU, CNN, CNN-LSTM, CapsNet
- 3 fractal methods: None, Hurst, Wavelet
- All feature combinations (15 total)

### 2. Optimized Analysis (8-12 hours)

```python
# Fast analysis with optimized configuration
python optimized_analysis.py
```

**Optimizations:**
- 3 core indicators: DIX, GEX, SKEW
- 4 main models: ARIMA, ANN, LSTM, GRU
- 2 fractal methods: None, Hurst
- Reduced epochs (15) and simplified architectures

## Data Requirements

### Input Data Format

```csv
DATE,VIX,DIX,GEX,SKEW,PUTCALLRATIO
2012-10-05,15.23,45.2,1200,120,0.85
2012-10-06,16.45,44.8,1150,118,0.87
...
```

### Market Indicators

- **VIX**: CBOE Volatility Index (target variable)
- **DIX**: Dark Index - dark pool activity indicator
- **GEX**: Gamma Exposure Index - options market positioning
- **SKEW**: CBOE SKEW Index - tail risk measure
- **PUTCALLRATIO**: Put/Call ratio - sentiment indicator

### Market Regimes

```python
market_periods = {
    'bull': ('2012-10-05', '2018-01-01'),     # Bull market
    'bear': ('2018-01-02', '2020-03-16'),     # Bear market + COVID crisis
    'recovery': ('2020-03-17', '2025-03-27')  # Recovery period
}
```

## Technical Implementation

### Fractal Analysis Methods

#### 1. Hurst Exponent

```python
def hurst_exponent(ts):
    """
    Calculate Hurst exponent for persistence analysis
    H > 0.5: Persistent (trending)
    H < 0.5: Anti-persistent (mean-reverting)
    H = 0.5: Random walk
    """
    lags = range(2, 20)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]  # Hurst exponent
```

#### 2. Wavelet Energy Features

```python
def apply_wavelet_energy(segment, wavelet='db4', level=3):
    """
    Extract energy features from wavelet coefficients
    Captures both frequency and time information
    """
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    energy = [np.sum(c**2) for c in coeffs]
    return energy
```

### Deep Learning Architectures

#### LSTM Configuration

```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=input_shape),
    LSTM(64),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss='mse')
```

#### CNN-LSTM Hybrid

```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(1)
])
```

### Statistical Validation

#### Diebold-Mariano Test

```python
def diebold_mariano_test(y_true, y_pred1, y_pred2):
    """
    Test statistical significance of forecasting differences
    H0: Models have equal predictive accuracy
    H1: Models have different predictive accuracy
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    d_t = e1**2 - e2**2
    
    mean_d = np.mean(d_t)
    var_d = np.var(d_t, ddof=1)
    n = len(d_t)
    
    DM_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - norm.cdf(np.abs(DM_stat)))
    
    return DM_stat, p_value
```

## Results Analysis

### Key Performance Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error  
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

### Expected Performance Improvements

- **Fractal Enhancement**: 15-35% MSE reduction
- **Deep Learning vs Traditional**: 3-4 orders of magnitude improvement
- **Statistical Significance**: 72% of comparisons significant (p<0.05)

### Regime-Specific Insights

- **Bull Market**: Consistent but modest improvements (10-20%)
- **Bear Market**: Moderate benefits during transitions
- **Recovery Period**: Maximum fractal enhancement (up to 35%)

## Checkpoint System

Both scripts include robust checkpoint systems for long-running analyses:

```python
# Save progress every 5 combinations
save_checkpoint_fast(results, predictions, regime_summaries, 
                    current_regime, combo_idx, total_combos)

# Resume from latest checkpoint
resume, checkpoint_data = should_resume_from_checkpoint_fast()
```

## Output Files

### Generated Results

```
results/
├── comprehensive_results.csv          # Full analysis results
├── optimized_results.csv             # Fast analysis results  
├── all_predictions.csv               # Detailed predictions
├── dm_results.csv                    # Diebold-Mariano test results
├── regime_characteristics.csv        # Market regime statistics
├── model_parameters.txt              # Model configurations
└── checkpoint_*.pkl                  # Progress checkpoints
```

### Visualization Outputs

```
plots/
├── all_models_performance.png        # Performance comparison
├── regime_analysis/                  # Regime-specific plots
├── dm_test_analysis.png             # Statistical validation
└── market_condition_analysis.png    # Volatility regime analysis
```

## Configuration Options

### Model Selection

```python
# Comprehensive analysis
model_types = ['ARIMA', 'GARCH', 'ANN', 'RNN', 'LSTM', 
               'GRU', 'CNN', 'CNN_LSTM', 'CapsNet']

# Optimized analysis  
ml_models = ['ANN', 'LSTM', 'GRU']
statistical_models = ['ARIMA']
```

### Feature Engineering

```python
# Base indicators
base_cols = ['DIX', 'GEX', 'SKEW', 'PUTCALLRATIO']

# Fractal methods
fractal_methods = ['none', 'hurst', 'wavelet']

# All combinations (1 to 4 features)
feature_combinations = list(itertools.combinations(base_cols, r))
```

### Hyperparameter Optimization

```python
param_grid_optimized = {
    'ANN': {
        'layers': [[64,32], [32,16]],
        'learning_rate': [1e-3],
        'batch_size': [32]
    }
}
```

## Production Recommendations

Based on empirical results:

1. **Architecture**: Prioritize LSTM/GRU models
2. **Features**: Use Hurst exponent preprocessing  
3. **Regime Adaptation**: Implement dynamic calibration
4. **Multi-Indicator**: Combine DIX+GEX+SKEW+PUTCALLRATIO
5. **Validation**: Apply time-series cross-validation

## Scientific Contributions

- **Methodological**: Integration of fractal analysis with deep learning
- **Empirical**: Multi-regime performance evaluation framework
- **Practical**: Production-ready optimization guidelines
- **Theoretical**: Evidence for fractal market hypothesis in volatility forecasting

## Important Notes

### Computational Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **CPU**: Multi-core processor recommended
- **GPU**: Optional but accelerates deep learning training
- **Storage**: ~2GB for full analysis outputs

### Data Quality

- Ensure continuous time series (no gaps)
- Handle missing values appropriately
- Verify data alignment across indicators
- Check for outliers and anomalies

### Risk Disclaimers

- Past performance does not guarantee future results
- Models should be validated on out-of-sample data
- Consider regime changes and structural breaks
- Regular model recalibration recommended

## Support & Citation

For questions, issues, or collaboration opportunities, please refer to the academic paper:

> "Enhancing Financial Time Series Prediction: Integrating Fractal Filters and Deep Learning Architectures"

### Key References

- Mandelbrot, B. B. (1963). The variation of certain speculative prices
- Peters, E. E. (1994). Fractal Market Analysis
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy

---

*This framework represents a cutting-edge approach to financial volatility forecasting, combining theoretical rigor with practical applicability for quantitative finance applications.*
