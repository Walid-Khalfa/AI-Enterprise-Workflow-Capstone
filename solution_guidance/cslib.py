#!/usr/bin/env python
"""
collection of functions for the final case study solution
"""

import os
import sys
import re
import shutil
import time
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from pandas.plotting import register_matplotlib_converters # Deprecated since pandas 0.25
# register_matplotlib_converters() # No longer needed if using matplotlib >= 3.2

COLORS = ["darkorange","royalblue","slategrey"]

def train_model(data):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score # Not used in this function for regression 'score'
    
    X = data[['price', 'times_viewed', 'stream_id']]
    X['stream_id'] = pd.to_numeric(X['stream_id'], errors='coerce')
    X.fillna(0, inplace=True) # Add this line to fill NaNs
    y = data['price'] # Assuming 'price' is the target for this example model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    # predictions = model.predict(X_test) # Not used
    accuracy = model.score(X_test, y_test) # For LinearRegression, score is R^2
    
    return model, accuracy

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def predict(model, X):
    """
    Generate predictions using a trained model
    """
    return model.predict(X)


def fetch_data(data_dir):
    """
    load all json formatted files into a dataframe
    """

    ## input testing
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist")
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data dir does not contain any files")

    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search(r"\.json",f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    ## read data into a temp structure
    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    ## ensure the data are formatted with correct columns
    for f,df_loop in all_months.items(): # Renamed df to df_loop to avoid conflict
        cols = set(df_loop.columns.tolist())
        if 'StreamID' in cols:
             df_loop.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in cols:
            df_loop.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in cols:
            df_loop.rename(columns={'total_price':'price'},inplace=True)

        current_cols = df_loop.columns.tolist() # Use current_cols after renaming
        if sorted(current_cols) != correct_columns:
            # Provide more info for debugging
            missing = set(correct_columns) - set(current_cols)
            extra = set(current_cols) - set(correct_columns)
            raise Exception(f"Columns mismatch for file {f}. Missing: {missing}, Extra: {extra}. Expected: {correct_columns}, Got: {sorted(current_cols)}")


    ## concat all of the data
    df_concat = pd.concat(list(all_months.values()),sort=True) # Renamed df to df_concat
    years,months,days = df_concat['year'].values,df_concat['month'].values,df_concat['day'].values 
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df_concat.shape[0])]
    df_concat['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    df_concat['invoice'] = [re.sub(r"\D+","",i) if isinstance(i, str) else str(i) for i in df_concat['invoice'].values] # Handle non-string invoices
    
    ## sort by date and reset the index
    df_concat.sort_values(by='invoice_date',inplace=True)
    df_concat.reset_index(drop=True,inplace=True)
    
    return(df_concat)


def convert_to_ts(df_orig, country=None):
    """
    given the original DataFrame (fetch_data())
    return a numerically indexed time-series DataFrame 
    by aggregating over each day
    """

    if country:
        if country not in np.unique(df_orig['country'].values):
            raise Exception("country {} not found in dataframe".format(country)) # Corrected typo and added country to message
    
        mask = df_orig['country'] == country
        df = df_orig[mask].copy() # Use .copy() to avoid SettingWithCopyWarning later if df is modified
    else:
        df = df_orig.copy() # Use .copy()
        
    ## use a date range to ensure all days are accounted for in the data
    if df.empty: # Handle empty dataframe for a country
        print(f"Warning: No data for country '{country if country else 'all'}'. Returning empty DataFrame.")
        return pd.DataFrame({'date':[],'purchases':[],'unique_invoices':[],
                             'unique_streams':[],'total_views':[],'year_month':[],'revenue':[]})

    invoice_dates = df['invoice_date'].values # This is already datetime64[D] from fetch_data
    
    # Ensure date operations are robust, especially if df is empty or has few records
    start_date_calc = df['invoice_date'].min()
    end_date_calc = df['invoice_date'].max()
    
    # Create a full date range from min to max date in the data
    days = pd.date_range(start_date_calc, end_date_calc, freq='D').values.astype('datetime64[D]')
    
    # Aggregate data for each day in the 'days' range
    # Group by date and aggregate first, then reindex with the full 'days' range
    
    # Ensure 'invoice_date' is the index for efficient grouping or merging
    df_grouped = df.groupby('invoice_date').agg(
        purchases=('invoice', 'size'), # Count of purchases (rows)
        unique_invoices=('invoice', 'nunique'),
        unique_streams=('stream_id', 'nunique'),
        total_views=('times_viewed', 'sum'),
        revenue=('price', 'sum')
    ).reset_index()

    # Create a DataFrame with the full 'days' range
    df_time_base = pd.DataFrame({'date': days})
    
    # Merge with aggregated data
    df_time = pd.merge(df_time_base, df_grouped, on='date', how='left').fillna(0)
    
    # Add year_month
    df_time['year_month'] = df_time['date'].astype('datetime64[M]').astype(str)


    # Cast columns to appropriate types, especially integer for counts
    int_cols = ['purchases', 'unique_invoices', 'unique_streams', 'total_views']
    for col in int_cols:
        df_time[col] = df_time[col].astype(int)
        
    return(df_time)


def fetch_ts(data_dir, clean=False):
    """
    convenience function to read in new data
    uses csv to load quickly
    use clean=True when you want to re-create the files
    """

    ts_data_dir = os.path.join(data_dir,"ts-data")
    
    if clean and os.path.exists(ts_data_dir): # Check existence before rmtree
        print(f"... cleaning old data from {ts_data_dir}")
        shutil.rmtree(ts_data_dir)
    
    if not os.path.exists(ts_data_dir):
        try:
            os.makedirs(ts_data_dir) # Use makedirs to create parent dirs if necessary
        except OSError as e:
            print(f"Error creating directory {ts_data_dir}: {e}")
            # Decide how to handle this: raise error or return empty
            return {}


    ## if files have already been processed load them        
    if len(os.listdir(ts_data_dir)) > 0 and not clean: # Ensure not clean before loading
        print("... loading ts data from files")
        loaded_data = {}
        for cf in os.listdir(ts_data_dir):
            if cf.startswith("ts-") and cf.endswith(".csv"):
                key = cf[3:-4] # Remove "ts-" and ".csv"
                try:
                    loaded_data[key] = pd.read_csv(os.path.join(ts_data_dir,cf), parse_dates=['date'])
                except Exception as e:
                    print(f"Error loading {cf}: {e}")
        if loaded_data: # If any data was successfully loaded
            return loaded_data


    ## get original data
    print("... processing data for loading")
    try:
        df_raw = fetch_data(data_dir) # Renamed df to df_raw
    except Exception as e:
        print(f"Error in fetch_data: {e}. Cannot proceed with ts conversion.")
        return {}


    ## find the top ten countries (wrt revenue)
    # Handle empty df_raw gracefully
    if df_raw.empty:
        print("Warning: Fetched raw data is empty. Cannot determine top countries or create time series.")
        return {}
        
    table = pd.pivot_table(df_raw,index='country',values="price",aggfunc='sum')
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue',inplace=True,ascending=False)
    top_ten_countries =  np.array(list(table.index))[:10]

    # file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search(r"\.json$",f)] # Unused
    # countries = [os.path.join(data_dir,"ts-"+re.sub(r"\s+","_",c.lower()) + ".csv") for c in top_ten_countries] # Unused

    ## load the data
    dfs_ts = {} # Renamed dfs to dfs_ts
    dfs_ts['all'] = convert_to_ts(df_raw, country=None) # Pass None explicitly for 'all'
    for country_name in top_ten_countries: # Renamed country to country_name
        country_id = re.sub(r"\s+","_",country_name.lower())
        # file_name = os.path.join(data_dir,"ts-"+ country_id + ".csv") # This was for saving, not needed here for key
        dfs_ts[country_id] = convert_to_ts(df_raw,country=country_name) # Pass original country_name

    ## save the data as csvs    
    for key, item_df in dfs_ts.items(): # Renamed item to item_df
        if not item_df.empty: # Only save if dataframe is not empty
            try:
                item_df.to_csv(os.path.join(ts_data_dir,"ts-"+key+".csv"),index=False)
            except Exception as e:
                print(f"Error saving ts data for {key}: {e}")
        
    return(dfs_ts)

def engineer_features(df,training=True):
    """
    for any given day the target becomes the sum of the next days revenue
    for that day we engineer several features that help predict the summed revenue
    
    the 'training' flag will trim data that should not be used for training
    when set to false all data will be returned

    """

    if df.empty or 'date' not in df.columns:
        print("Warning: DataFrame is empty or 'date' column is missing in engineer_features. Returning empty structures.")
        return pd.DataFrame(), np.array([]), np.array([])

    ## extract dates
    # Ensure 'date' is datetime64[D]
    dates = pd.to_datetime(df['date']).values.astype('datetime64[D]')


    ## engineer some features
    eng_features = defaultdict(list)
    previous_windows =[7, 14, 28, 70]  # Renamed 'previous' to 'previous_windows'
    y = np.zeros(dates.size)
    
    # Pre-calculate sums for unique_invoices and total_views for efficiency if df is large
    # This requires df to be sorted by date, which it should be from convert_to_ts
    # For simplicity, keeping original loop structure for now.

    for d,day_dt64 in enumerate(dates): # Renamed day to day_dt64

        ## use windows in time back from a specific date
        current_dt64 = day_dt64 # Already datetime64[D]
        for num in previous_windows:
            prev_dt64 = current_dt64 - np.timedelta64(num, 'D')
            # Replace np.in1d with np.isin
            mask = np.isin(dates, np.arange(prev_dt64, current_dt64, dtype='datetime64[D]'))
            eng_features["previous_{}".format(num)].append(df[mask]['revenue'].sum())

        ## get get the target revenue    
        plus_30_dt64 = current_dt64 + np.timedelta64(30,'D')
        # Replace np.in1d with np.isin
        mask = np.isin(dates, np.arange(current_dt64, plus_30_dt64, dtype='datetime64[D]'))
        y[d] = df[mask]['revenue'].sum()

        ## attempt to capture monthly trend with previous years data (if present)
        start_date_prev_year = current_dt64 - np.timedelta64(365,'D')
        stop_date_prev_year = plus_30_dt64 - np.timedelta64(365,'D')
        # Replace np.in1d with np.isin
        mask = np.isin(dates, np.arange(start_date_prev_year, stop_date_prev_year, dtype='datetime64[D]'))
        eng_features['previous_year'].append(df[mask]['revenue'].sum())

        ## add some non-revenue features
        minus_30_dt64 = current_dt64 - np.timedelta64(30,'D')
        # Replace np.in1d with np.isin
        mask = np.isin(dates, np.arange(minus_30_dt64, current_dt64, dtype='datetime64[D]'))
        # Handle cases where mask results in empty slice (e.g., early in data)
        eng_features['recent_invoices'].append(df[mask]['unique_invoices'].mean() if mask.any() else 0)
        eng_features['recent_views'].append(df[mask]['total_views'].mean() if mask.any() else 0)


    X = pd.DataFrame(eng_features)
    ## combine features in to df and remove rows with all zeros
    X.fillna(0,inplace=True) # Keep one fillna
    
    # Ensure y and dates are pandas Series/arrays for consistent masking
    y_series = pd.Series(y)
    dates_series = pd.Series(dates)

    mask_sum_gt_zero = X.sum(axis=1) > 0
    X = X[mask_sum_gt_zero]
    y_series = y_series[mask_sum_gt_zero]
    dates_series = dates_series[mask_sum_gt_zero]
    
    X.reset_index(drop=True, inplace=True)
    y_final = y_series.values # Convert back to numpy array
    dates_final = dates_series.values # Convert back to numpy array


    if training == True:
        ## remove the last 30 days (because the target is not reliable)
        if X.shape[0] > 30 : # Ensure there are enough rows to remove last 30
            mask_training = np.arange(X.shape[0]) < (X.shape[0] - 30) # More direct way to get all but last 30
            X = X[mask_training]
            y_final = y_final[mask_training]
            dates_final = dates_final[mask_training]
            X.reset_index(drop=True, inplace=True) # Reset index again after this mask
        else:
            print("Warning: Not enough data for training after filtering (less than 30 days). Consider more data or different filtering.")
            # Depending on policy, could return empty or as is
            # For now, returns potentially small X, y, dates
    
    return(X,y_final,dates_final)


# Modular data processing
# Reusable across training and production
if __name__ == "__main__":

    run_start = time.time() 
    # Changed data_dir to be relative to project root, assuming script is run from there.
    data_dir = os.path.join("data","cs-train") 
    print("...fetching data")

    ts_all = fetch_ts(data_dir,clean=False)

    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))

    if ts_all: # Check if ts_all is not empty
        for key,item in ts_all.items():
            print(key,item.shape)
            # Example of engineering features for 'all' data
            if key == 'all':
                X_example, y_example, dates_example = engineer_features(item, training=True)
                print(f"Engineered features for 'all': X shape {X_example.shape}, y shape {y_example.shape}, dates shape {dates_example.shape}")

    # Génération de graphiques de comparaison (commented out as it's incomplete)
    # def generate_performance_plot(baseline, current):
    #     plt.plot(baseline, label='Base')
    #     plt.plot(current, label='Modèle')
    # # Avant
    # # Après