# Day 2: Iris Dataset Exploration

## What I Did
- ✓ Loaded Iris dataset (150 flowers, 4 features)
- ✓ Explored features: SepalLength, SepalWidth, PetalLength, PetalWidth
- ✓ Checked data quality: No missing values!
- ✓ Analyzed distributions with box plots
- ✓ Visualized relationships with scatter plots
- ✓ Computed correlations with heatmap

## Key Findings

### 1. Balanced Dataset
- 50 flowers of each species (Setosa, Versicolor, Virginica)
- Perfect for supervised learning!

### 2. Quality Data
- Zero missing values
- All features have good variation
- No data cleaning needed

### 3. Clear Patterns
- Box plots show species have different measurements
- Scatter plots show three distinct clusters
- Easy to classify!

## Connection to Day 1 Concepts

### Supervised Learning ✓
- We have LABELED data
- 150 examples with correct flower types
- Perfect for training a classifier

### Features ✓
- 4 measurements describe each flower
- SepalLength, SepalWidth, PetalLength, PetalWidth
- All features are informative

### Data Quality ✓
- No overfitting concerns yet (plenty of clean data)
- Balanced classes (no bias toward one type)

## Visualizations Created
- Box plots: Feature distributions by species
- Scatter plots: Feature relationships
- Heatmap: Feature correlations

## Next Steps (Day 3)
1. Split data: 70% train, 30% test
2. Build classification model
3. Train and evaluate
4. Check for overfitting
5. Calculate metrics (accuracy, precision, recall, F1)

## Files
- `01-data-exploration.ipynb` - Complete analysis with code and plots
