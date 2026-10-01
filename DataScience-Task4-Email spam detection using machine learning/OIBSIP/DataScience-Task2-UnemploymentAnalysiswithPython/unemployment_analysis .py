# Unemployment Analysis EDA - Jupyter Notebook Content
# This file contains the complete code for unemployment analysis
# Run this in Jupyter Notebook (.ipynb)

"""
================================================================================
UNEMPLOYMENT ANALYSIS IN INDIA: EXPLORATORY DATA ANALYSIS
Task 2 - Data Science Track | OASIS Infobyte
================================================================================
Objective: Perform exploratory data analysis on unemployment data to uncover 
regional and temporal trends, with a focus on COVID-19 pandemic impact.
================================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND INITIAL SETUP
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("UNEMPLOYMENT ANALYSIS IN INDIA")
print("Exploratory Data Analysis (EDA)")
print("=" * 80)

# ============================================================================
# SECTION 2: DATA LOADING AND INITIAL INSPECTION
# ============================================================================

"""
INSTRUCTIONS FOR DATA DOWNLOAD:
1. Go to https://www.kaggle.com/datasets
2. Search for "Unemployment in India"
3. Download the dataset (usually named 'Unemployment_in_India.csv')
4. Place it in your working directory
"""

# Load the dataset
df = pd.read_csv('Unemployment_in_India.csv')

# Fix messy headers: strip stray whitespace, then rename to short, consistent names
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Labour Participation Rate (%)': 'Estimated Labour Participation Rate'
})
# Strip stray whitespace from text columns too (common in this dataset)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

print("\n" + "=" * 80)
print("SECTION 1: INITIAL DATA INSPECTION")
print("=" * 80)

print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names and Types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nNull Values Count:")
print(df.isnull().sum())
print(f"\nBasic Statistics:")
print(df.describe())

# Data Cleaning
print("\n" + "=" * 80)
print("SECTION 2: DATA CLEANING")
print("=" * 80)

# Check for duplicate rows
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Handle missing values
print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows with missing critical values
df_clean = df.dropna(subset=['Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate'])
print(f"\nRows after removing critical nulls: {df_clean.shape[0]}")

# Convert Date column to datetime if it exists
if 'Date' in df_clean.columns:
    df_clean['Date'] = pd.to_datetime(df_clean['Date'].str.strip(), dayfirst=True)
    df_clean = df_clean.sort_values('Date')
    print(f"Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")

print("\nCleaned dataset info:")
print(df_clean.info())

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# 3.1 UNEMPLOYMENT RATE STATISTICS
print("\n--- UNEMPLOYMENT RATE STATISTICS ---")
print(f"Mean Unemployment Rate: {df_clean['Unemployment Rate'].mean():.2f}%")
print(f"Median Unemployment Rate: {df_clean['Unemployment Rate'].median():.2f}%")
print(f"Std Dev: {df_clean['Unemployment Rate'].std():.2f}%")
print(f"Min: {df_clean['Unemployment Rate'].min():.2f}%")
print(f"Max: {df_clean['Unemployment Rate'].max():.2f}%")

# 3.2 REGION-WISE ANALYSIS
print("\n--- REGION-WISE UNEMPLOYMENT ANALYSIS ---")
if 'Region' in df_clean.columns:
    region_stats = df_clean.groupby('Region').agg({
        'Unemployment Rate': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)
    print("\nUnemployment Rate by Region:")
    print(region_stats)
    
    # Sort by average unemployment
    region_avg = df_clean.groupby('Region')['Unemployment Rate'].mean().sort_values(ascending=False)
    print("\nRegions ranked by average unemployment (highest to lowest):")
    print(region_avg)

# 3.3 STATE-WISE ANALYSIS (if available)
if 'State' in df_clean.columns or 'Region' in df_clean.columns:
    state_col = 'State' if 'State' in df_clean.columns else 'Region'
    print(f"\n--- TOP 10 STATES WITH HIGHEST AVERAGE UNEMPLOYMENT ---")
    top_10_states = df_clean.groupby(state_col)['Unemployment Rate'].mean().sort_values(ascending=False).head(10)
    print(top_10_states)

# 3.4 EMPLOYMENT STATISTICS
print("\n--- EMPLOYMENT STATISTICS ---")
if 'Estimated Employed' in df_clean.columns:
    print(f"Mean Employed: {df_clean['Estimated Employed'].mean():.0f}")
    print(f"Total Employment across dataset: {df_clean['Estimated Employed'].sum():.0f}")

# 3.5 LABOUR PARTICIPATION ANALYSIS
print("\n--- LABOUR PARTICIPATION RATE ANALYSIS ---")
if 'Estimated Labour Participation Rate' in df_clean.columns:
    print(f"Mean Labour Participation Rate: {df_clean['Estimated Labour Participation Rate'].mean():.2f}%")
    print(f"Median: {df_clean['Estimated Labour Participation Rate'].median():.2f}%")
    print(f"Std Dev: {df_clean['Estimated Labour Participation Rate'].std():.2f}%")

# ============================================================================
# SECTION 4: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: VISUALIZATIONS")
print("=" * 80)

# 4.1 TIME SERIES: UNEMPLOYMENT RATE OVER TIME
print("\n[VISUALIZATION 1] Time Series: Unemployment Rate Over Time")
fig, ax = plt.subplots(figsize=(16, 6))

if 'Date' in df_clean.columns:
    # Get top 5 regions/states
    top_regions = df_clean.groupby('Region')['Unemployment Rate'].mean().nlargest(5).index
    
    for region in top_regions:
        region_data = df_clean[df_clean['Region'] == region].sort_values('Date')
        ax.plot(region_data['Date'], region_data['Unemployment Rate'], 
                marker='o', label=region, linewidth=2, markersize=4)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Unemployment Rate Trends Over Time (Top 5 Regions)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("✓ Time series visualization created")

# 4.2 BAR CHART: TOP 10 STATES WITH HIGHEST UNEMPLOYMENT
print("\n[VISUALIZATION 2] Bar Chart: Top 10 States with Highest Average Unemployment")
fig, ax = plt.subplots(figsize=(12, 6))

state_col = 'State' if 'State' in df_clean.columns else 'Region'
top_10 = df_clean.groupby(state_col)['Unemployment Rate'].mean().sort_values(ascending=False).head(10)

bars = ax.barh(range(len(top_10)), top_10.values, color='#FF6B6B')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10.index)
ax.set_xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 States/Regions with Highest Average Unemployment Rate', 
             fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (idx, val) in enumerate(top_10.items()):
    ax.text(val, i, f' {val:.2f}%', va='center', fontweight='bold')

ax.invert_yaxis()
plt.tight_layout()
plt.show()

print("✓ Top 10 states bar chart created")

# 4.3 DISTRIBUTION OF UNEMPLOYMENT RATES
print("\n[VISUALIZATION 3] Distribution of Unemployment Rates")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df_clean['Unemployment Rate'], bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Unemployment Rate (%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Unemployment Rates', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Box plot by region (if available)
if 'Region' in df_clean.columns:
    region_order = df_clean.groupby('Region')['Unemployment Rate'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_clean, y='Region', x='Unemployment Rate', 
                order=region_order, ax=axes[1], palette='Set2')
    axes[1].set_xlabel('Unemployment Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Region', fontsize=11, fontweight='bold')
    axes[1].set_title('Unemployment Rate Distribution by Region', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("✓ Distribution visualizations created")

# 4.4 CORRELATION HEATMAP
print("\n[VISUALIZATION 4] Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))

# Select numeric columns for correlation
numeric_cols = ['Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']
numeric_cols = [col for col in numeric_cols if col in df_clean.columns]

if len(numeric_cols) > 1:
    corr_matrix = df_clean[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix: Key Unemployment Metrics', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    print("✓ Correlation heatmap created")
    print("\nCorrelation Analysis:")
    print(corr_matrix)

# 4.5 MONTHLY TRENDS
print("\n[VISUALIZATION 5] Monthly Unemployment Trends")
if 'Date' in df_clean.columns:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    monthly_data = df_clean.groupby(df_clean['Date'].dt.to_period('M'))['Unemployment Rate'].agg(['mean', 'std'])
    monthly_data.index = monthly_data.index.to_timestamp()
    
    ax.plot(monthly_data.index, monthly_data['mean'], marker='o', linewidth=2, 
            label='Average', color='#2E86AB', markersize=6)
    ax.fill_between(monthly_data.index, 
                     monthly_data['mean'] - monthly_data['std'],
                     monthly_data['mean'] + monthly_data['std'],
                     alpha=0.2, color='#2E86AB', label='±1 Std Dev')
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Average Unemployment Rate with Standard Deviation', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("✓ Monthly trends visualization created")

# ============================================================================
# SECTION 5: COVID-19 IMPACT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: COVID-19 PANDEMIC IMPACT ANALYSIS")
print("=" * 80)

"""
Key Note: COVID-19 outbreak in India
- First case: January 30, 2020
- Lockdown Phase 1: March 25 - April 14, 2020
- Extended lockdowns: April-May 2020
- We'll compare data before March 2020 (pre-COVID) vs. after March 2020 (COVID period)
"""

if 'Date' in df_clean.columns:
    covid_date = pd.to_datetime('2020-03-01')
    
    pre_covid = df_clean[df_clean['Date'] < covid_date]
    covid_period = df_clean[df_clean['Date'] >= covid_date]
    
    print(f"\nPre-COVID Period: {pre_covid['Date'].min().date()} to {pre_covid['Date'].max().date()}")
    print(f"COVID Period: {covid_period['Date'].min().date()} to {covid_period['Date'].max().date()}")
    
    # 5.1 PRE-COVID vs COVID COMPARISON
    print("\n" + "-" * 80)
    print("PRE-COVID vs COVID UNEMPLOYMENT STATISTICS")
    print("-" * 80)
    
    comparison_metrics = ['Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']
    comparison_metrics = [col for col in comparison_metrics if col in df_clean.columns]
    
    print(f"\n{'Metric':<40} {'Pre-COVID':<15} {'COVID Period':<15} {'Change':<15}")
    print("-" * 80)
    
    for metric in comparison_metrics:
        pre_avg = pre_covid[metric].mean()
        covid_avg = covid_period[metric].mean()
        change = covid_avg - pre_avg
        change_pct = (change / pre_avg * 100) if pre_avg != 0 else 0
        
        print(f"{metric:<40} {pre_avg:<15.2f} {covid_avg:<15.2f} {change_pct:>13.2f}%")
    
    # 5.2 COVID IMPACT VISUALIZATION
    print("\n[VISUALIZATION 6] COVID-19 Impact: Before vs During Pandemic")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Unemployment rate comparison
    categories = ['Pre-COVID\n(Before Mar 2020)', 'COVID Period\n(Mar 2020 onwards)']
    unemployment_values = [pre_covid['Unemployment Rate'].mean(), covid_period['Unemployment Rate'].mean()]
    colors = ['#2ECC71', '#E74C3C']
    
    bars = axes[0].bar(categories, unemployment_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Average Unemployment Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Unemployment Rate: Pre-COVID vs COVID Period', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(unemployment_values) * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, unemployment_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Regional comparison during COVID
    if 'Region' in df_clean.columns:
        region_covid_comparison = covid_period.groupby('Region')['Unemployment Rate'].mean().sort_values(ascending=False)
        axes[1].barh(range(len(region_covid_comparison)), region_covid_comparison.values, color='#E67E22', alpha=0.7)
        axes[1].set_yticks(range(len(region_covid_comparison)))
        axes[1].set_yticklabels(region_covid_comparison.index)
        axes[1].set_xlabel('Average Unemployment Rate (%)', fontsize=11, fontweight='bold')
        axes[1].set_title('Regional Unemployment During COVID-19 Period', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ COVID-19 impact visualization created")
    
    # 5.3 STATE-WISE COVID IMPACT
    print("\n[VISUALIZATION 7] State-wise COVID Impact on Unemployment")
    if 'State' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        state_col = 'State'
        state_impact = []
        
        for state in df_clean[state_col].unique():
            state_pre = pre_covid[pre_covid[state_col] == state]['Unemployment Rate'].mean()
            state_covid = covid_period[covid_period[state_col] == state]['Unemployment Rate'].mean()
            
            if not pd.isna(state_pre) and not pd.isna(state_covid):
                impact = state_covid - state_pre
                state_impact.append({'State': state, 'Impact': impact})
        
        if state_impact:
            impact_df = pd.DataFrame(state_impact).sort_values('Impact', ascending=False)
            
            colors_impact = ['#E74C3C' if x > 0 else '#2ECC71' for x in impact_df['Impact']]
            bars = ax.barh(range(len(impact_df)), impact_df['Impact'].values, color=colors_impact, alpha=0.7)
            ax.set_yticks(range(len(impact_df)))
            ax.set_yticklabels(impact_df['State'].values)
            ax.set_xlabel('Change in Unemployment Rate (% points)', fontsize=11, fontweight='bold')
            ax.set_title('COVID-19 Impact on Unemployment by State\n(Positive = Increase, Negative = Decrease)', 
                        fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.show()
            
            print("✓ State-wise COVID impact visualization created")

# ============================================================================
# SECTION 6: KEY INSIGHTS AND CONCLUSIONS
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: KEY INSIGHTS AND CONCLUSIONS")
print("=" * 80)

insights = """
1. OVERALL UNEMPLOYMENT TRENDS:
   - India's unemployment rate showed significant variation across regions
   - Peak unemployment rate: {max_rate:.2f}%
   - Lowest unemployment rate: {min_rate:.2f}%
   
2. REGIONAL DISPARITIES:
   - Highest unemployment regions require targeted policy interventions
   - Regional differences suggest varied economic structures and recovery rates
   
3. COVID-19 PANDEMIC IMPACT:
   - The pandemic significantly affected employment patterns
   - Difference in unemployment rate (Pre-COVID vs COVID): {covid_diff:.2f} percentage points
   - Urban and rural sectors were impacted differently
   
4. LABOUR PARTICIPATION:
   - Labour force participation rate shows correlation with unemployment
   - Economic activity and employment opportunities vary by region
   
5. EMPLOYMENT STATISTICS:
   - Total employment figures show workforce size and changes
   - Regional economic capacity affects job creation
   
RECOMMENDATIONS FOR FURTHER ANALYSIS:
   ✓ Analyze sector-wise unemployment (agriculture, manufacturing, services)
   ✓ Study age-group and gender-wise unemployment disparities
   ✓ Correlation with economic indicators (GDP, investments)
   ✓ Policy impact analysis during lockdown phases
"""

max_rate = df_clean['Unemployment Rate'].max()
min_rate = df_clean['Unemployment Rate'].min()
covid_diff = covid_period['Unemployment Rate'].mean() - pre_covid['Unemployment Rate'].mean() if 'Date' in df_clean.columns else 0

print(insights.format(max_rate=max_rate, min_rate=min_rate, covid_diff=covid_diff))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nDataset used: Unemployment_in_India.csv")
print(f"Records analyzed: {len(df_clean)}")
print(f"Date range: {df_clean['Date'].min().date() if 'Date' in df_clean.columns else 'N/A'} to {df_clean['Date'].max().date() if 'Date' in df_clean.columns else 'N/A'}")
