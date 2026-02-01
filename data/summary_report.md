
# Model Performance Summary Report

## Overall Performance
- **Test KL Divergence Loss**: 0.054825
- **Overall Mean Absolute Error**: 0.042929
- **Overall Mean Squared Error**: 0.004246

## Cell Line Performance Analysis

| Cell Line | Correlation | MAE | MSE | Mean True | Mean Predicted |
|-----------|-------------|-----|-----|-----------|----------------|
| GM12878 | 0.5671 | 0.0629 | 0.0072 | 0.3219 | 0.3260 |
| H1Esc | 0.5887 | 0.0649 | 0.0077 | 0.4346 | 0.4381 |
| HAP1 | 0.5278 | 0.0378 | 0.0028 | 0.1020 | 0.1013 |
| HFF | 0.5584 | 0.0411 | 0.0033 | 0.1345 | 0.1272 |
| IMR90 | 0.1101 | 0.0080 | 0.0001 | 0.0070 | 0.0075 |

## Key Findings

### Best Performing Cell Lines (by MAE):
1. **IMR90**: MAE = 0.0080, Correlation = 0.1101
2. **HAP1**: MAE = 0.0378, Correlation = 0.5278
3. **HFF**: MAE = 0.0411, Correlation = 0.5584

### Worst Performing Cell Lines (by MAE):
1. **H1Esc**: MAE = 0.0649, Correlation = 0.5887
2. **GM12878**: MAE = 0.0629, Correlation = 0.5671
3. **HFF**: MAE = 0.0411, Correlation = 0.5584

## Biological Insights

- **Dominant Cell Line Prediction Accuracy**: 0.777 (77.7%)
- **Average Prediction Entropy**: 1.2378
- **Average True Entropy**: 1.1900

## Model Configuration
- **Training Epochs**: 42
- **Best Validation Loss**: 0.057351
- **Final Training Loss**: 0.054911
