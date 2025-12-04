"""
Script for running workforce diversity analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


print("Workforce Diversity Analysis Report")

print("\nLoading data")

# Load data
df = pd.read_csv('dataset.csv')

print(f"\nDataset Overview:")
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['year'].dropna().unique())}")
print(f"Number of sectors: {df['sector'].nunique()}")

# Data cleaning and preprocessing
df_clean = df.dropna(subset=['year', 'sector']).copy()
df_clean = df_clean[df_clean['sector'] != 'Total, 16 years and over']

percent_cols = ['percent_women', 'percent_white', 'percent_black_or_african_american', 
                'percent_asian', 'percent_hispanic_or_latino']
ethnic_cols = ['percent_white', 'percent_black_or_african_american', 
               'percent_asian', 'percent_hispanic_or_latino']

# Fix data type issues
for col in percent_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    if df_clean[col].max() > 1:
        df_clean.loc[df_clean[col] > 1, col] = df_clean.loc[df_clean[col] > 1, col] / 100

df_clean = df_clean.dropna(subset=percent_cols)

print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Sectors in analysis: {df_clean['sector'].nunique()}")


# Q1: Which sectors have similar demographic structures?


print("Q1: Sectors with Similar Demographic Structures")


sector_demographics = df_clean.groupby('sector')[percent_cols + ['total_employed_in_thousands']].mean()

# Prepare data for clustering
X = sector_demographics[percent_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')

# Create dendrogram
plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, labels=sector_demographics.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering of Sectors by Demographic Structure', fontsize=14, fontweight='bold')
plt.xlabel('Sector', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig('sector_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] sector_clustering_dendrogram.png")
plt.close()

# K-means clustering
silhouette_scores = []
K_range = range(2, min(8, len(sector_demographics)))
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")
print(f"Silhouette score: {max(silhouette_scores):.3f}")

# Apply K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
sector_demographics['cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
sector_demographics['PC1'] = X_pca[:, 0]
sector_demographics['PC2'] = X_pca[:, 1]

# Visualize clusters
plt.figure(figsize=(14, 10))
colors = sns.color_palette('husl', optimal_k)
for i in range(optimal_k):
    cluster_data = sector_demographics[sector_demographics['cluster'] == i]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                c=[colors[i]], s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                label=f'Cluster {i+1}')
    
    for idx, row in cluster_data.iterrows():
        plt.annotate(idx, (row['PC1'], row['PC2']), 
                    fontsize=8, ha='center', va='bottom', 
                    xytext=(0, 5), textcoords='offset points')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('Sector Clusters by Demographic Composition', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sector_clusters_pca.png', dpi=300, bbox_inches='tight')
print("[SAVED] sector_clusters_pca.png")
plt.close()

# Display cluster compositions
print("\nCluster Compositions:")
for i in range(optimal_k):
    print(f"\n--- Cluster {i+1} ---")
    cluster_sectors = sector_demographics[sector_demographics['cluster'] == i]
    print(f"Sectors: {', '.join(cluster_sectors.index)}")
    print(f"\nAverage Demographics:")
    for col in percent_cols:
        print(f"  {col}: {cluster_sectors[col].mean()*100:.1f}%")
    print(f"Avg Workforce Size: {cluster_sectors['total_employed_in_thousands'].mean():.0f}k")


# Q2: Gender imbalance and relationship to workforce size


print("Q2: Gender Imbalance and Workforce Size")


sector_gender = df_clean.groupby('sector').agg({
    'percent_women': 'mean',
    'total_employed_in_thousands': 'mean'
}).reset_index()

sector_gender['gender_imbalance'] = abs(sector_gender['percent_women'] - 0.5)
sector_gender['dominant_gender'] = sector_gender['percent_women'].apply(
    lambda x: 'Female' if x > 0.5 else 'Male'
)

# Create a mapping of full sector names to abbreviated labels
sector_abbreviations = {
    'Agriculture, forestry, fishing, and hunting': 'AGRI',
    'Mining, quarrying, and oil and gas extraction': 'MINE',
    'Construction': 'CONS',
    'Manufacturing': 'MANF',
    'Wholesale and retail trade': 'TRAD',
    'Transportation and utilities': 'TRAN',
    'Information': 'INFO',
    'Financial activities': 'FINA',
    'Professional and business services': 'PROF',
    'Education and health services': 'EDUC',
    'Leisure and hospitality': 'LEIS',
    'Other services': 'OTHR',
    'Public administration': 'PUBL'
}

# Add abbreviated labels
sector_gender['sector_abbrev'] = sector_gender['sector'].map(sector_abbreviations)

sector_gender_sorted = sector_gender.sort_values('gender_imbalance', ascending=False)

print("\nGender Imbalance by Sector:")
print(sector_gender_sorted[['sector', 'percent_women', 'gender_imbalance', 
                             'dominant_gender', 'total_employed_in_thousands']].to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 14))

# Top plot: Gender distribution
ax1 = axes[0]
sector_gender_sorted_plot = sector_gender_sorted.copy()
sector_gender_sorted_plot['percent_men'] = 1 - sector_gender_sorted_plot['percent_women']

x = np.arange(len(sector_gender_sorted_plot))
ax1.barh(x, sector_gender_sorted_plot['percent_women'], 
         label='Women', color='#FF6B6B', alpha=0.8)
ax1.barh(x, sector_gender_sorted_plot['percent_men'], 
         left=sector_gender_sorted_plot['percent_women'],
         label='Men', color='#4ECDC4', alpha=0.8)
ax1.set_yticks(x)
ax1.set_yticklabels(sector_gender_sorted_plot['sector'], fontsize=9)
ax1.set_xlabel('Gender Composition', fontsize=12)
ax1.set_title('Gender Distribution by Sector', fontsize=14, fontweight='bold')
ax1.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Bottom plot: Improved imbalance vs size with abbreviations
ax2 = axes[1]
colors_scatter = sector_gender['dominant_gender'].map({'Female': '#FF6B6B', 'Male': '#4ECDC4'})

# Create the scatter plot with abbreviated labels
scatter = ax2.scatter(sector_gender['total_employed_in_thousands'], 
                     sector_gender['gender_imbalance']*100,
                     s=sector_gender['total_employed_in_thousands']*3,  # Slightly smaller bubbles
                     c=colors_scatter, alpha=0.7, edgecolors='black', linewidth=1)

# Use abbreviated labels for annotations
for idx, row in sector_gender.iterrows():
    ax2.annotate(row['sector_abbrev'], 
                (row['total_employed_in_thousands'], row['gender_imbalance']*100),
                fontsize=10, ha='center', va='center', fontweight='bold')

ax2.set_xlabel('Total Workforce Size (thousands)', fontsize=12)
ax2.set_ylabel('Gender Imbalance (% deviation from 50-50)', fontsize=12)
ax2.set_title('Gender Imbalance vs Workforce Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Create legend for sector abbreviations
legend_text = []
for full_name, abbrev in sector_abbreviations.items():
    if full_name in sector_gender['sector'].values:
        legend_text.append(f'{abbrev}: {full_name}')

# Add legend box outside the plot area
legend_box = '\n'.join(legend_text)
ax2.text(1.02, 0.5, legend_box, transform=ax2.transAxes, fontsize=9,
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add color legend for gender dominance
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#FF6B6B', alpha=0.7, label='Female-dominated'),
                  Patch(facecolor='#4ECDC4', alpha=0.7, label='Male-dominated')]
ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Make room for the legend
plt.savefig('gender_imbalance_analysis.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] gender_imbalance_analysis.png")
plt.close()

correlation = sector_gender['total_employed_in_thousands'].corr(sector_gender['gender_imbalance'])
print(f"\nCorrelation between workforce size and gender imbalance: {correlation:.3f}")


# Q3: Diversity shifts 2021-2023


print("Q3: Diversity Shifts (2021 -> 2023)")


df_2021 = df_clean[df_clean['year'] == 2021].groupby('sector')[percent_cols].mean()
df_2023 = df_clean[df_clean['year'] == 2023].groupby('sector')[percent_cols].mean()

diversity_changes = pd.DataFrame()
for col in percent_cols:
    diversity_changes[f'{col}_change'] = df_2023[col] - df_2021[col]

diversity_changes['total_shift'] = np.sqrt((diversity_changes**2).sum(axis=1))
diversity_changes = diversity_changes.sort_values('total_shift', ascending=False)

print("\nTop Sectors by Total Diversity Shift:")
print(diversity_changes[['total_shift']].head(10))

print("\n\nDetailed Changes (in percentage points):")
for sector in diversity_changes.head(5).index:
    print(f"\n{sector}:")
    print(f"  Women: {diversity_changes.loc[sector, 'percent_women_change']*100:+.2f}pp")
    print(f"  White: {diversity_changes.loc[sector, 'percent_white_change']*100:+.2f}pp")
    print(f"  Black/African American: {diversity_changes.loc[sector, 'percent_black_or_african_american_change']*100:+.2f}pp")
    print(f"  Asian: {diversity_changes.loc[sector, 'percent_asian_change']*100:+.2f}pp")
    print(f"  Hispanic/Latino: {diversity_changes.loc[sector, 'percent_hispanic_or_latino_change']*100:+.2f}pp")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Workforce Diversity Shifts (2021 → 2023)', fontsize=16, fontweight='bold')

metrics = [
    ('percent_women_change', 'Gender (% Women)'),
    ('percent_black_or_african_american_change', 'Black/African American'),
    ('percent_asian_change', 'Asian'),
    ('percent_hispanic_or_latino_change', 'Hispanic/Latino')
]

for idx, (metric, label) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    data = diversity_changes[metric].sort_values()
    colors_bar = ['red' if x < 0 else 'green' for x in data.values]
    data.plot(kind='barh', ax=ax, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Change (percentage points)', fontsize=11)
    ax.set_title(f'{label} Change', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('diversity_shifts_2021_2023.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] diversity_shifts_2021_2023.png")
plt.close()


# Q4: Ethnic dominance


print("Q4: Ethnic Dominance by Sector")


sector_ethnicity = df_clean.groupby('sector')[ethnic_cols + ['total_employed_in_thousands']].mean()

sector_ethnicity['dominant_group'] = sector_ethnicity[ethnic_cols].idxmax(axis=1)
sector_ethnicity['dominance_level'] = sector_ethnicity[ethnic_cols].max(axis=1)
sector_ethnicity['HHI'] = (sector_ethnicity[ethnic_cols]**2).sum(axis=1)

sector_ethnicity['dominant_group'] = sector_ethnicity['dominant_group'].str.replace('percent_', '').str.replace('_', ' ').str.title()

sector_ethnicity_sorted = sector_ethnicity.sort_values('dominance_level', ascending=False)

print("\nHHI > 0.5 = High concentration, 0.25-0.5 = Moderate, < 0.25 = Low\n")
print(sector_ethnicity_sorted[['dominant_group', 'dominance_level', 'HHI', 
                                'total_employed_in_thousands']].to_string())

extreme_threshold = 0.80
extreme_dominance = sector_ethnicity_sorted[sector_ethnicity_sorted['dominance_level'] > extreme_threshold]
print(f"\n\nSectors with EXTREME ethnic dominance (>{extreme_threshold*100}%):")
for sector, row in extreme_dominance.iterrows():
    print(f"  - {sector}: {row['dominance_level']*100:.1f}% {row['dominant_group']}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Stacked bar chart
ax1 = axes[0]
ethnic_data = sector_ethnicity_sorted[ethnic_cols]
ethnic_data.columns = [col.replace('percent_', '').replace('_', ' ').title() 
                       for col in ethnic_data.columns]

ethnic_data.plot(kind='barh', stacked=True, ax=ax1, 
                 color=['#E8E8E8', '#2C3E50', '#F39C12', '#16A085'],
                 alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Ethnic Composition', fontsize=12)
ax1.set_title('Ethnic Distribution by Sector', fontsize=14, fontweight='bold')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)

# HHI chart
ax2 = axes[1]
hhi_sorted = sector_ethnicity.sort_values('HHI', ascending=True)
colors_hhi = ['red' if x > 0.5 else 'orange' if x > 0.25 else 'green' 
              for x in hhi_sorted['HHI']]
hhi_sorted['HHI'].plot(kind='barh', ax=ax2, color=colors_hhi, 
                       alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('HHI (Concentration Index)', fontsize=12)
ax2.set_title('Ethnic Concentration (HHI)', fontsize=14, fontweight='bold')
ax2.axvline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='High')
ax2.axvline(0.25, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Moderate')
ax2.legend(fontsize=9)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ethnic_dominance_analysis.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] ethnic_dominance_analysis.png")
plt.close()


# Q5: Ethnic diversity and sector size


print("Q5: Ethnic Diversity and Sector Size")


def shannon_diversity(row):
    proportions = np.array([row[col] for col in ethnic_cols])
    proportions = proportions[proportions > 0]
    if len(proportions) == 0:
        return 0
    return -np.sum(proportions * np.log(proportions))

sector_ethnicity['shannon_diversity'] = sector_ethnicity.apply(shannon_diversity, axis=1)
sector_ethnicity['diversity_score'] = 1 - sector_ethnicity['HHI']

diversity_analysis = sector_ethnicity[['shannon_diversity', 'diversity_score', 
                                       'total_employed_in_thousands']].copy()
# Add abbreviated labels for sectors
diversity_analysis['sector_abbrev'] = diversity_analysis.index.map(sector_abbreviations)
diversity_analysis = diversity_analysis.sort_values('shannon_diversity', ascending=False)

print("\nShannon Diversity Index: Higher = more diverse\n")
print(diversity_analysis.to_string())

correlation_shannon = diversity_analysis['total_employed_in_thousands'].corr(
    diversity_analysis['shannon_diversity']
)

print(f"\n\nCorrelation between sector size and Shannon diversity: {correlation_shannon:.3f}")

print("\n\nMost Diverse Sectors:")
for sector in diversity_analysis.head(3).index:
    print(f"  {sector}:")
    print(f"    Shannon Index: {diversity_analysis.loc[sector, 'shannon_diversity']:.3f}")
    print(f"    Workforce: {diversity_analysis.loc[sector, 'total_employed_in_thousands']:.0f}k")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Diversity ranking
ax1 = axes[0]
diversity_sorted = diversity_analysis.sort_values('shannon_diversity', ascending=True)
colors_div = plt.cm.RdYlGn(diversity_sorted['shannon_diversity'] / diversity_sorted['shannon_diversity'].max())
diversity_sorted['shannon_diversity'].plot(kind='barh', ax=ax1, color=colors_div, 
                                            edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Shannon Diversity Index', fontsize=12)
ax1.set_title('Ethnic Diversity by Sector', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Diversity vs size
ax2 = axes[1]
scatter = ax2.scatter(diversity_analysis['total_employed_in_thousands'],
                     diversity_analysis['shannon_diversity'],
                     s=diversity_analysis['total_employed_in_thousands']*3,  # Slightly smaller bubbles
                     c=diversity_analysis['shannon_diversity'],
                     cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1)

# Use abbreviated labels for annotations
for idx, row in diversity_analysis.iterrows():
    ax2.annotate(row['sector_abbrev'], 
                (row['total_employed_in_thousands'], row['shannon_diversity']),
                fontsize=10, ha='center', va='center', fontweight='bold')

# Trend line
z = np.polyfit(diversity_analysis['total_employed_in_thousands'], 
               diversity_analysis['shannon_diversity'], 1)
p = np.poly1d(z)
x_trend = np.linspace(diversity_analysis['total_employed_in_thousands'].min(),
                      diversity_analysis['total_employed_in_thousands'].max(), 100)
ax2.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, 
         label=f'Trend (r={correlation_shannon:.3f})')

ax2.set_xlabel('Total Workforce Size (thousands)', fontsize=12)
ax2.set_ylabel('Shannon Diversity Index', fontsize=12)
ax2.set_title('Ethnic Diversity vs Workforce Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Create legend for sector abbreviations
legend_text = []
for full_name, abbrev in sector_abbreviations.items():
    if full_name in diversity_analysis.index:
        legend_text.append(f'{abbrev}: {full_name}')

# Add legend box at the bottom of the plot
# Arrange in columns for better space usage
n_cols = 4
legend_items = [legend_text[i:i+n_cols] for i in range(0, len(legend_text), n_cols)]
legend_box = '\n'.join(['  |  '.join(col) for col in legend_items])
ax2.text(0.5, -0.15, legend_box, transform=ax2.transAxes, fontsize=8,
         horizontalalignment='center', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Add trend line legend
ax2.legend(fontsize=10, loc='upper right')

# Add colorbar
plt.colorbar(scatter, ax=ax2, label='Shannon Diversity')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for the legend at the bottom
plt.savefig('ethnic_diversity_analysis.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] ethnic_diversity_analysis.png")
plt.close()


# Q6: Group industries by ethnic composition


print("Q6: Industry Grouping by Ethnic Composition")


# Since sector-level already provides grouping, use hierarchical clustering
# on ethnic composition specifically
X_ethnic = sector_ethnicity[ethnic_cols].values
scaler_ethnic = StandardScaler()
X_ethnic_scaled = scaler_ethnic.fit_transform(X_ethnic)

linkage_ethnic = linkage(X_ethnic_scaled, method='ward')

plt.figure(figsize=(14, 10))
dendrogram(linkage_ethnic, labels=sector_ethnicity.index, 
          leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering of Sectors by Ethnic Composition', 
         fontsize=14, fontweight='bold')
plt.xlabel('Sector', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig('sector_ethnic_clustering.png', dpi=300, bbox_inches='tight')
print("\n[SAVED] sector_ethnic_clustering.png")
plt.close()

# K-means on ethnic composition
silhouette_scores_ethnic = []
K_range_ethnic = range(2, min(7, len(sector_ethnicity)))
for k in K_range_ethnic:
    kmeans_ethnic = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_ethnic = kmeans_ethnic.fit_predict(X_ethnic_scaled)
    silhouette_scores_ethnic.append(silhouette_score(X_ethnic_scaled, labels_ethnic))

optimal_k_ethnic = K_range_ethnic[np.argmax(silhouette_scores_ethnic)]
kmeans_ethnic = KMeans(n_clusters=optimal_k_ethnic, random_state=42, n_init=10)
sector_ethnicity['ethnic_cluster'] = kmeans_ethnic.fit_predict(X_ethnic_scaled)

print(f"\nIdentified {optimal_k_ethnic} ethnic composition clusters\n")
for i in range(optimal_k_ethnic):
    print(f"\n--- Ethnic Cluster {i+1} ---")
    cluster_sectors = sector_ethnicity[sector_ethnicity['ethnic_cluster'] == i]
    print(f"Sectors: {', '.join(cluster_sectors.index)}")
    print(f"\nAverage Ethnic Composition:")
    for col in ethnic_cols:
        print(f"  {col.replace('percent_', '').title()}: {cluster_sectors[col].mean()*100:.1f}%")


# Executive Summary & Key Findings


print("Executive Summary & Key Findings")


print("\n1. Demographic Clustering")
print("-" * 40)
print(f"   - Identified {optimal_k} distinct sector clusters")
print("   - Clustering based on gender AND ethnic composition")

print("\n2. Gender Imbalance")
print("-" * 40)
most_male = sector_gender_sorted[sector_gender_sorted['dominant_gender'] == 'Male'].iloc[0]
most_female = sector_gender_sorted[sector_gender_sorted['dominant_gender'] == 'Female'].iloc[0]
print(f"   - Most male-dominated: {most_male['sector']} ({most_male['percent_women']*100:.1f}% women)")
print(f"   - Most female-dominated: {most_female['sector']} ({most_female['percent_women']*100:.1f}% women)")
print(f"   - Correlation with size: {correlation:.3f}")

print("\n3. Diversity Trends (2021-2023)")
print("-" * 40)
print(f"   - Greatest shift: {diversity_changes.index[0]}")
print(f"   - Magnitude: {diversity_changes.iloc[0]['total_shift']:.4f}")

print("\n4. Ethnic Concentration")
print("-" * 40)
print(f"   - Most concentrated: {sector_ethnicity_sorted.index[0]}")
print(f"   - {len(extreme_dominance)} sectors with >80% single group")

print("\n5. Ethnic Diversity")
print("-" * 40)
print(f"   - Most diverse: {diversity_analysis.index[0]}")
print(f"   - Least diverse: {diversity_analysis.index[-1]}")
print(f"   - Size correlation: {correlation_shannon:.3f}")

print("\n6. Ethnic Grouping")
print("-" * 40)
print(f"   - {optimal_k_ethnic} distinct ethnic composition patterns")


print("Policy Recommendations")


print("\n[High Priority]:")
extreme_gender = sector_gender_sorted[sector_gender_sorted['gender_imbalance'] > 0.35]
for idx, row in extreme_gender.head(3).iterrows():
    print(f"   - {row['sector']}: {row['percent_women']*100:.1f}% women")

print("\n[Medium Priority]:")
for sector in extreme_dominance.head(3).index:
    print(f"   - {sector}: Ethnic concentration")

# Export summary
summary_table = sector_demographics[['percent_women', 'total_employed_in_thousands']].copy()
summary_table['gender_imbalance'] = sector_gender.set_index('sector')['gender_imbalance']
summary_table['dominant_ethnic_group'] = sector_ethnicity['dominant_group']
summary_table['ethnic_dominance'] = sector_ethnicity['dominance_level']
summary_table['ethnic_diversity_shannon'] = sector_ethnicity['shannon_diversity']
summary_table['HHI'] = sector_ethnicity['HHI']
summary_table['cluster'] = sector_demographics['cluster']
summary_table['ethnic_cluster'] = sector_ethnicity['ethnic_cluster']

summary_table.to_csv('sector_diversity_summary.csv')
diversity_changes.to_csv('diversity_changes_2021_2023.csv')


print("Analysis Complete")

#print("\nGenerated files:")
#print("  - sector_clustering_dendrogram.png")
#print("  - sector_clusters_pca.png")
#print("  - gender_imbalance_analysis.png")
#print("  - diversity_shifts_2021_2023.png")
#print("  - ethnic_dominance_analysis.png")
#print("  - ethnic_diversity_analysis.png")
#print("  - sector_ethnic_clustering.png")
#print("  - sector_diversity_summary.csv")
#print("  - diversity_changes_2021_2023.csv")


