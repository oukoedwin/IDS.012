import numpy as np
import pandas as pd

# TODO: add a feature for the number of generators owned by the owner of this generator
# TODO: add a feature for winter vs summer difference in bid prices for each generator

# Firm column name (i.e the firm that owns the generator): "Masked Lead Participant ID"
# Date column name: "Day" and is in the format MM/DD/YYYY (useful in detecting winter from summer. 
# Assume October to March is winter and April to September is summer)

def process_generator_data(df):
    df['Day'] = pd.to_datetime(df['Day'])
    df['is_winter'] = df['Day'].dt.month.between(10, 12) | df['Day'].dt.month.between(1, 3)
    # Group by generator type
    grouped = df.groupby('Masked Asset ID')
    
    # Initialize result dataframe
    result = pd.DataFrame(index=grouped.groups.keys())

    # Owner counts: for Hypothesis Testing
    owner_counts = df.groupby('Masked Lead Participant ID')['Masked Asset ID'].nunique()
    result['Owner Generator Count'] = df.groupby('Masked Asset ID')['Masked Lead Participant ID'].first().map(owner_counts).fillna(0)
    
    # Ensure we have at least one day per generator
    total_days = grouped.size()
    valid_generators = total_days > 0
    result = result[valid_generators]
    grouped = df[df['Masked Asset ID'].isin(result.index)].groupby('Masked Asset ID')
    total_days = total_days[valid_generators]
    
    # Low Bidder
    segment_cols = [f"Segment {i} Price" for i in range(1, 11)]
    min_bids = df[segment_cols].min(axis=1) 
    low_bid_days = (min_bids <= 5).groupby(df['Masked Asset ID']).sum()
    result['Low Bidder'] = (low_bid_days / total_days).fillna(0) >= 0.95  # Fill NaN with 0
    
    #  Average Bid Start
    result['Average Bid Start'] = grouped['Segment 1 Price'].mean()
    
    # Capacity metrics
    econ_max = grouped['Economic Maximum']
    result['Generation capacity'] = econ_max.max()
    result['Capacity Dispersion'] = econ_max.var().fillna(0)  # Fill NaN with 0 for single-value groups
    result['Capacity Range'] = econ_max.max() - econ_max.min()
    
    # Status ratios
    status_counts = df.groupby(['Masked Asset ID', 'Unit Status']).size().unstack(fill_value=0)
    total_counts = status_counts.sum(axis=1).replace(0, 1)  # Avoid division by zero
    result['EconRatio'] = status_counts.get('ECONOMIC', pd.Series(0, index=result.index)) / total_counts
    result['EmerRatio'] = status_counts.get('EMERGENCY', pd.Series(0, index=result.index)) / total_counts
    result['MustRunRatio'] = status_counts.get('MUST_RUN', pd.Series(0, index=result.index)) / total_counts
    
    # Diurnal Bidder
    df['is_night'] = (df['Trading Interval'] > 18) | (df['Trading Interval'] <= 6)
    night_bids = df[df['is_night'] & (df['Economic Maximum'] < 0.1)].groupby('Masked Asset ID').size()
    day_bids = df[~df['is_night'] & (df['Economic Maximum'] > 0.1)].groupby('Masked Asset ID').size()
    ratio = (night_bids.add(day_bids, fill_value=0)) / total_counts.replace(0, 1)  # Avoid division by zero
    result['Diurnal Bidder'] = ratio.fillna(0) >= 0.95  # Fill NaN with 0
    
    # Average Marginal Cost Slope
    def calc_marginal_slope(g):
        last_price = g[segment_cols].apply(lambda x: x[x.last_valid_index()], axis=1)
        first_price = g[segment_cols].apply(lambda x: x[x.first_valid_index()], axis=1)
        denominator = (g['Economic Maximum'] - g['Economic Minimum']).replace(0, 1)  # Avoid division by zero
        return ((last_price - first_price) / denominator).mean()
    
    result['Average Marginal Cost Slope'] = grouped.apply(calc_marginal_slope).fillna(0)
    
    #  Price metrics
    result['No load price'] = grouped['No Load Price'].mean()
    result['cold startup price'] = grouped['Cold Startup Price'].mean()
    result['intermediate startup cost'] = grouped['Intermediate Startup Price'].mean()
    result['hot startup cost'] = grouped['Hot Startup Price'].mean()
    
    #  Reservation Market Bidder
    claim_days = df[df['Claim 10'].notna() | df['Claim 30'].notna()].groupby('Masked Asset ID').size()
    result['Reservation Market Bidder'] = (claim_days / total_days.replace(0, 1)).fillna(0) >= 0.75
    
    #  Implied Maximum Ramp Rate
    max_claim10 = (grouped['Claim 10'].max() / 10).replace([np.inf, -np.inf], np.nan).fillna(0)
    max_claim30 = (grouped['Claim 30'].max() / 30).replace([np.inf, -np.inf], np.nan).fillna(0)
    result['Implied Maximum Ramp Rate'] = np.maximum(max_claim10, max_claim30)
    
    # Maximum Daily Energy Bidder
    max_bids = df[segment_cols].max(axis=1).groupby(df['Masked Asset ID']).max()
    result['Maximum Daily Energy Bidder'] = max_bids.fillna(0) > 0

    # Winter-Summer Bid Difference: For Hypothesis Testing

    def calc_seasonal_diff(g):
        winter_avg = g[g['is_winter']][segment_cols].mean().mean()
        summer_avg = g[~g['is_winter']][segment_cols].mean().mean()
        return winter_avg - summer_avg
    result['Winter-Summer Bid Diff'] = grouped.apply(calc_seasonal_diff).fillna(0)
    
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
    "Maximum Daily Energy Bidder", "Owner Generator Count", "Winter-Summer Bid Diff"
]
generator_df = generator_df[column_names]
print(generator_df[["Average Bid Start"]].describe())
generator_df.to_csv("./data/generators.csv")