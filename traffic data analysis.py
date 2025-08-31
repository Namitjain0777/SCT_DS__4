import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Updated file path
file_path = r"C:\Users\Lenovo\Downloads\df.csv"

print("Loading traffic accident data...")
try:
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Basic data exploration
print("\n" + "="*50)
print("DATA OVERVIEW")
print("="*50)
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
missing_data = df.isnull().sum()
missing_summary = missing_data[missing_data > 0].sort_values(ascending=False)
if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("No missing values found!")

# Data types summary
print(f"\nData types:")
print(df.dtypes.value_counts())

# Convert datetime columns if they exist
potential_datetime_cols = ['Start_Time', 'End_Time', 'Date', 'Timestamp', 'DateTime']
datetime_cols_found = [col for col in potential_datetime_cols if col in df.columns]

for col in datetime_cols_found:
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"Converted {col} to datetime")
    except:
        print(f"Could not convert {col} to datetime")

# Extract time features from the first datetime column found
datetime_col = None
for col in datetime_cols_found:
    if df[col].dtype == 'datetime64[ns]':
        datetime_col = col
        break

if datetime_col:
    print(f"Using {datetime_col} for temporal analysis")
    df['Hour'] = df[datetime_col].dt.hour
    df['Day_of_Week'] = df[datetime_col].dt.day_name()
    df['Month'] = df[datetime_col].dt.month
    df['Year'] = df[datetime_col].dt.year
    df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Fall', 10: 'Fall', 11: 'Fall'})

print("\n" + "="*50)
print("STATISTICAL SUMMARY")
print("="*50)
print(df.describe())

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

print("\n" + "="*50)
print("VISUALIZATION ANALYSIS")
print("="*50)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 16))

# 3. Top categorical variable analysis
plt.subplot(3, 4, 3)
if categorical_cols:
    # Find the categorical column with most variety
    cat_col = None
    for col in categorical_cols:
        if df[col].nunique() > 1 and df[col].nunique() < 50:
            cat_col = col
            break
    
    if cat_col:
        top_categories = df[cat_col].value_counts().head(8)
        plt.barh(range(len(top_categories)), top_categories.values, color='orange', alpha=0.8)
        plt.title(f'Top Categories: {cat_col}', fontsize=12, fontweight='bold')
        plt.xlabel('Count')
        plt.yticks(range(len(top_categories)), [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in top_categories.index])
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No suitable\ncategorical data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Categorical Analysis', fontsize=12)
else:
    plt.text(0.5, 0.5, 'No categorical\ndata available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Categorical Analysis - No Data', fontsize=12)

# 4. Numerical distribution analysis
plt.subplot(3, 4, )
if numerical_cols:
    # Choose first numerical column for distribution
    num_col = numerical_cols[0]
    data = df[num_col].dropna()
    if len(data) > 0:
        plt.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title(f'Distribution: {num_col}', fontsize=12, fontweight='bold')
        plt.xlabel(num_col)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, f'No data in\n{num_col}', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{num_col} - No Data', fontsize=12)
else:
    plt.text(0.5, 0.5, 'No numerical\ndata available', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Numerical Analysis - No Data', fontsize=12)

# 5-8. Additional analysis based on available columns
subplot_count = 5
for i, col in enumerate(categorical_cols[:4]):
    if subplot_count > 8:
        break
    
    plt.subplot(3, 4, subplot_count)

    if df[col].nunique() <= 20:  # Only plot if reasonable number of categories
            top_values = df[col].value_counts().head(10)
            if len(top_values) > 0:
                colors = plt.cm.Set3(np.linspace(0, 1, len(top_values)))
                plt.bar(range(len(top_values)), top_values.values, color=colors, alpha=0.8)
                plt.title(f'{col} Distribution', fontsize=12, fontweight='bold')
                plt.xlabel('Categories')
                plt.ylabel('Count')
                labels = [str(x)[:10] + '...' if len(str(x)) > 10 else str(x) for x in top_values.index]
                plt.xticks(range(len(top_values)), labels, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, f'No data in\n{col}', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{col} - No Data', fontsize=12)
 
plt.tight_layout()
plt.savefig('traffic_accident_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("HOTSPOT ANALYSIS")
print("="*50)

# Geographic hotspot analysis - check for various lat/lng column names
lat_cols = ['Start_Lat', 'Lat', 'Latitude', 'lat', 'latitude']
lng_cols = ['Start_Lng', 'Lng', 'Longitude', 'Long', 'lon', 'longitude']

lat_col = None
lng_col = None

for col in lat_cols:
    if col in df.columns:
        lat_col = col
        break

for col in lng_cols:
    if col in df.columns:
        lng_col = col
        break

if lat_col and lng_col:
    print(f"Found geographic columns: {lat_col}, {lng_col}")
    
    # Remove invalid coordinates
    geo_df = df[(df[lat_col].notna()) & (df[lng_col].notna())].copy()
    
    # Filter for reasonable US coordinates (adjust if data is from other regions)
    geo_df = geo_df[(geo_df[lat_col].between(-90, 90)) & 
                    (geo_df[lng_col].between(-180, 180))].copy()
    
    print(f"Valid geographic data points: {len(geo_df)}")
    
    if len(geo_df) > 0:
        # Sample data for performance if dataset is too large
        if len(geo_df) > 15000:
            geo_sample = geo_df.sample(n=15000, random_state=42)
            print("Sampling 15,000 points for visualization performance")
        else:
            geo_sample = geo_df
        
        # Find a good color column
        color_col = None
        severity_cols = ['Severity', 'severity', 'Accident_Severity']
        for col in severity_cols:
            if col in df.columns:
                color_col = col
                break
        
        if not color_col and categorical_cols:
            # Use first categorical column with reasonable number of categories
            for col in categorical_cols:
                if geo_sample[col].nunique() <= 10:
                    color_col = col
                    break
        
print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)


# Data quality summary
print(f"\nData Quality Summary:")
print(f"- Total records: {len(df):,}")
print(f"- Columns: {len(df.columns)}")
print(f"- Missing value percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")

# Create summary statistics table
if numerical_cols:
    print(f"\nNumerical Variables Summary:")
    summary_stats = df[numerical_cols].describe()
    print(summary_stats.round(2))

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("Generated files:")
print("- traffic_accident_analysis.png: Main analysis charts")
if len(numerical_cols) > 1:
    print("- correlation_matrix.png: Correlation heatmap")
if lat_col and lng_col:
    print("- Interactive map displayed (if successful)")
print(f"\nDataset contains {len(df):,} records with {len(df.columns)} variables")
print("Analysis reveals key patterns in accident data!")