# Workforce Diversity Analysis

This project analyzes workforce demographic patterns across sectors to identify diversity trends and clustering patterns.

## Installation

1. Install Python dependencies using pip:

```bash
pip install -r requirements.txt
```

Alternatively, install packages individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Running the Analysis

1. Ensure `dataset.csv` is in the same directory as the script.

2. Run the analysis script:

```bash
python workforce_diversity_analysis.py
```

The script will:
- Load and process the dataset
- Perform clustering and statistical analysis
- Generate PNG visualizations (saved in the current directory)
- Export CSV summary files
- Display analysis results in the console

## Output Files

The script generates the following files:
- PNG visualizations: Various charts and graphs analyzing diversity patterns
- CSV files: Summary data and diversity change metrics

## Project Structure

```
workforce_diversity/
├── dataset.csv                          # Original workforce demographic dataset
├── workforce_diversity_analysis.py      # Main analysis script
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── sector_clustering_dendrogram.png    # Hierarchical clustering visualization
├── sector_clusters_pca.png             # PCA cluster visualization
├── gender_imbalance_analysis.png       # Gender imbalance analysis charts
├── diversity_shifts_2021_2023.png     # Diversity change visualization
├── ethnic_dominance_analysis.png       # Ethnic concentration analysis
├── ethnic_diversity_analysis.png       # Ethnic diversity analysis
├── sector_ethnic_clustering.png        # Ethnic composition clustering
├── sector_diversity_summary.csv        # Summary metrics by sector
├── diversity_changes_2021_2023.csv     # Year-over-year diversity changes
├── Workforce_Diversity_Analysis_Report.tex  # LaTeX source document
└── Workforce_Diversity_Analysis_Report.pdf   # Compiled PDF report
```

