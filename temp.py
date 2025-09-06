import numpy as np
import pandas as pd

def process_generator_data(df):
    # Group by generator type
    grouped = df.groupby('Masked Asset ID')
    
    # Initialize result dataframe
    result = pd.DataFrame(index=grouped.groups.keys())
    
    # 1. Low Bidder - Generator bids <=$5 per MWh in 95% of the days
    segment_cols = [f"Segment {i} Price" for i in range(1, 11)]
    min_bids = df[segment_cols].min(axis=1) 
    low_bid_days = (min_bids <= 5).groupby(df['Masked Asset ID']).sum()
    total_days = grouped.size()
    result['Low Bidder'] = (low_bid_days / total_days) >= 0.95
    
    # 2. Average Bid Start
    result['Average Bid Start'] = grouped['Segment 1 Price'].mean()
    
    # 3. Generation capacity (max Economic Maximum)
    result['Generation capacity'] = grouped['Economic Maximum'].max()
    
    # 4. Capacity Dispersion (variance of Economic Maximum)
    result['Capacity Dispersion'] = grouped['Economic Maximum'].var()
    
    # 5. Capacity Range (max - min Economic Maximum)
    result['Capacity Range'] = grouped['Economic Maximum'].max() - grouped['Economic Maximum'].min()
    
    # 6-8. Status ratios
    status_counts = df.groupby(['Masked Asset ID', 'Unit Status']).size().unstack(fill_value=0)
    total_counts = status_counts.sum(axis=1)
    result['EconRatio'] = status_counts.get('ECONOMIC', pd.Series(0, index=result.index)) / total_counts
    result['EmerRatio'] = status_counts.get('EMERGENCY', pd.Series(0, index=result.index)) / total_counts
    result['MustRunRatio'] = status_counts.get('MUST_RUN', pd.Series(0, index=result.index)) / total_counts
    
    # 9. Diurnal Bidder
    df['is_night'] = (df['Trading Interval'] > 18) | (df['Trading Interval'] <= 6)
    night_bids = df[df['is_night'] & (df['Economic Maximum'] < 0.1)].groupby('Masked Asset ID').size()
    day_bids = df[~df['is_night'] & (df['Economic Maximum'] > 0.1)].groupby('Masked Asset ID').size()
    result['Diurnal Bidder'] = (night_bids.add(day_bids, fill_value=0) / total_counts) >= 0.95
    
    # 10. Average Marginal Cost Slope
    def calc_marginal_slope(g):
        last_price = g[segment_cols].apply(lambda x: x[x.last_valid_index()], axis=1)
        first_price = g[segment_cols].apply(lambda x: x[x.first_valid_index()], axis=1)
        return ((last_price - first_price) / 
                (g['Economic Maximum'] - g['Economic Minimum'])).mean()
    
    result['Average Marginal Cost Slope'] = grouped.apply(calc_marginal_slope)
    
    # 11-14. Price metrics
    result['No load price'] = grouped['No Load Price'].mean()
    result['cold startup price'] = grouped['Cold Startup Price'].mean()
    result['intermediate startup cost'] = grouped['Intermediate Startup Price'].mean()
    result['hot startup cost'] = grouped['Hot Startup Price'].mean()
    
    # 15. Reservation Market Bidder
    claim_days = df[df['Claim 10'].notna() | df['Claim 30'].notna()].groupby('Masked Asset ID').size()
    result['Reservation Market Bidder'] = (claim_days / total_days) >= 0.75
    
    # 16. Implied Maximum Ramp Rate
    max_claim10 = grouped['Claim 10'].max() / 10
    max_claim30 = grouped['Claim 30'].max() / 30
    result['Implied Maximum Ramp Rate'] = np.maximum(max_claim10, max_claim30)
    
    # 17. Maximum Daily Energy Bidder
    max_bids = df[segment_cols].max(axis=1).groupby(df['Masked Asset ID']).max()
    result['Maximum Daily Energy Bidder'] = max_bids > 0
    
    return result

# Load data and process
df = pd.read_csv("./data/combined.csv", header=0)
df['Trading Interval'] = pd.to_numeric(df['Trading Interval'], errors='coerce')
df = df[df['Trading Interval'].notna()]
generator_df = process_generator_data(df)

# Rearrange columns in this order
column_names = [
    "Low Bidder", "Average Bid Start", "Generation capacity", "Capacity Dispersion", "Capacity Range",
    "EconRatio", "EmerRatio", "MustRunRatio", "Diurnal Bidder", "Average Marginal Cost Slope",
    "No load price", "cold startup price", "intermediate startup cost", "hot startup cost", "Reservation Market Bidder", "Implied Maximum Ramp Rate",
    "Maximum Daily Energy Bidder"
]
generator_df = generator_df[column_names]
generator_df.to_csv("./data/generators.csv")