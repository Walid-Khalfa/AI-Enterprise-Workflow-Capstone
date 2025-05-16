import time
import os
import re
import joblib
import numpy as np
import argparse 
import pandas as pd # Import pandas for pd.Timestamp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from solution_guidance.logger import update_predict_log, update_train_log
from .cslib import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def _model_train(df,tag,test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 
    
    """

    ## start timer for runtime
    time_start = time.time()
    
    X,y,dates_engineered = engineer_features(df) # Renamed to avoid confusion

    if dates_engineered.size == 0: # Check if engineer_features returned empty dates
        print(f"Warning: No data returned from engineer_features for tag '{tag}'. Skipping training for this tag.")
        return

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        if n_samples == 0 and X.shape[0] > 0 : # ensure at least 1 sample if possible
             n_samples = 1
        if X.shape[0] == 0: # No data to sample from
            print(f"Warning: No data to sample for testing for tag '{tag}'. Skipping training.")
            return

        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.isin(np.arange(y.size),subset_indices) 
        y_subset=y[mask] # Renamed
        X_subset=X[mask] # Renamed
        dates_subset=dates_engineered[mask] # Renamed
        
        # Use these subsetted versions for training if test is True
        X_train_data, y_train_data, dates_log_data = X_subset, y_subset, dates_subset
    else:
        # Use full engineered data if not testing
        X_train_data, y_train_data, dates_log_data = X, y, dates_engineered


    if X_train_data.shape[0] == 0:
        print(f"Warning: No training data available for tag '{tag}' after potential subsetting. Skipping training.")
        return
    if dates_log_data.size == 0:
        print(f"Warning: No date data available for logging for tag '{tag}'. Skipping training (or logging date range).")
        # Decide whether to proceed without date range in log or skip entirely
        # For now, let's try to proceed but log will be affected.
        # A better check would be before this point.
        # If dates_log_data is empty here, it means X_train_data is also likely empty or an issue upstream.
        # The check above for X_train_data.shape[0] should catch this.

    ## Perform a train-test split
    # Use X_train_data and y_train_data for the split
    X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.25,
                                                        shuffle=True, random_state=42)
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Warning: Not enough data to perform train-test split for tag '{tag}'. Skipping training.")
        return

    ## train a random forest model
    param_grid_rf = {
    'rf__criterion': ['squared_error','absolute_error'], 
    'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor(random_state=42))]) 
    
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1) 
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    ## retrain using all data (X_train_data, y_train_data - which is either full or subset)
    grid.fit(X_train_data, y_train_data)
    model_name = re.sub(r"\.","_",str(MODEL_VERSION))
    
    # Ensure MODEL_DIR exists
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if test:
        saved_model_path = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model_path))
    else:
        saved_model_path = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model_path))
        
    joblib.dump(grid,saved_model_path)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    # Convert numpy.datetime64 to string 'YYYY-MM-DD'
    # Or convert to pandas Timestamp first then .date()
    if dates_log_data.size > 0:
        date_start_str = str(dates_log_data[0]) # numpy.datetime64 to 'YYYY-MM-DD' string
        date_end_str = str(dates_log_data[-1]) # numpy.datetime64 to 'YYYY-MM-DD' string
        # Alternative using pandas Timestamp if more manipulation is needed:
        # date_start_str = str(pd.Timestamp(dates_log_data[0]).date())
        # date_end_str = str(pd.Timestamp(dates_log_data[-1]).date())
    else:
        date_start_str = "N/A"
        date_end_str = "N/A"
        print(f"Warning: Date range for logging is N/A for tag '{tag}' due to empty dates_log_data.")

    update_train_log(tag,(date_start_str, date_end_str),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=test)
  

def model_train(data_dir,test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR) # Use makedirs

    if test:
        print("... test flag on")
        print("...... subsetting countries") 
        print("...... subsetting countries data") 
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)
    if not ts_data:
        print("Error: No time series data fetched. Aborting model training.")
        return

    ## train a different model for each data sets
    for country,df_country in ts_data.items():
        if df_country.empty:
            print(f"Warning: DataFrame for country '{country}' is empty. Skipping training for this country.")
            continue
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        print(f"... Training model for country: {country}")
        _model_train(df_country,country,test=test)
    
def model_load(prefix='sl', data_dir=None, training=True):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    'data_dir' is crucial for loading associated data for the models.
    'training' flag affects feature engineering for the data.
    """

    if not data_dir:
        data_dir = os.path.join("data", "cs-train")
    
    models_path = MODEL_DIR 
    if not os.path.isdir(models_path):
        if prefix == 'sl': 
            print(f"Warning: Models directory '{models_path}' not found. Assuming no models trained yet for prefix '{prefix}'.")
            return {}, {} 
        else:
            return {}, {}

    models = [f for f in os.listdir(models_path) if f.startswith(prefix+"-") and f.endswith(".joblib")]

    if not models:
        print(f"Warning: No models with prefix '{prefix}' found in '{models_path}'.")
        return {}, {}

    all_models = {}
    for model_file in models: 
        try:
            name_part = model_file[len(prefix)+1:-len(".joblib")] 
            country_key = name_part.split('-')[0] 
            all_models[country_key] = joblib.load(os.path.join(models_path,model_file))
        except Exception as e: 
            print(f"Warning: Could not parse or load model file name '{model_file}'. Error: {e}. Skipping.")
            continue

    ts_data = fetch_ts(data_dir) 
    if not ts_data:
        print(f"Warning: No time-series data fetched by model_load for data_dir '{data_dir}'. Returning loaded models without associated data.")
        return {}, all_models # Or handle as error depending on requirements

    all_data = {}
    for country, df_country in ts_data.items(): 
        if df_country.empty:
            print(f"Warning: DataFrame for country '{country}' is empty in model_load. Skipping feature engineering for this country.")
            all_data[country] = {"X":pd.DataFrame(),"y":np.array([]),"dates": np.array([])} # Store empty structures
            continue

        X,y,dates_pd = engineer_features(df_country,training=training) 
        if X.empty: # If engineer_features returns empty X
             all_data[country] = {"X":X,"y":y,"dates": dates_pd} # Store what was returned
             continue
        dates_str = np.array([str(d) for d in dates_pd]) # Convert numpy.datetime64 to string
        all_data[country] = {"X":X,"y":y,"dates": dates_str}
        
    return(all_data, all_models)


def model_predict(country,year,month,day,all_data=None, all_models=None,test=False, data_dir=None):
    time_start = time.time()

    if all_data is None or all_models is None or not all_models.get(country) or country not in all_data or all_data[country]["X"].empty:
        print("... Loading data and models for prediction (or specific country missing/empty).")
        current_data_dir = data_dir if data_dir else os.path.join("data", "cs-train")
        loaded_data, loaded_models = model_load(data_dir=current_data_dir, training=False)
        
        # Update all_data and all_models with potentially newly loaded ones
        # Be careful about overwriting if they were partially passed
        if all_data is None: all_data = {}
        if all_models is None: all_models = {}
        all_data.update(loaded_data)
        all_models.update(loaded_models)


    if country not in all_models:
        raise Exception(f"ERROR (model_predict) - model for country '{country}' could not be found after loading attempt.")
    if country not in all_data or all_data[country]["X"].empty:
         raise Exception(f"ERROR (model_predict) - data for country '{country}' is missing or empty after loading attempt.")

    for d_val in [year, month, day]:
        if not isinstance(d_val, str) or re.search(r"\D", d_val): 
            raise Exception("ERROR (model_predict) - invalid year, month, or day (must be strings of digits).")
    
    model_to_predict = all_models[country] # Renamed
    data_country = all_data[country]

    target_date_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    print(f"... Target date: {target_date_str}")

    if target_date_str not in data_country['dates']:
        raise Exception(
            f"ERROR (model_predict) - date {target_date_str} not in range "
            f"{data_country['dates'][0] if data_country['dates'].size > 0 else 'N/A'} - "
            f"{data_country['dates'][-1] if data_country['dates'].size > 0 else 'N/A'}"
        )
    
    date_indx = np.where(data_country['dates'] == target_date_str)[0][0]
    query = data_country['X'].iloc[[date_indx]]
    
    if data_country['dates'].shape[0] != data_country['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch between dates and X.")

    y_pred = model_to_predict.predict(query) # Use renamed variable
    y_proba = None
    if hasattr(model_to_predict, 'predict_proba'): 
        if callable(model_to_predict.predict_proba): 
             try:
                y_proba = model_to_predict.predict_proba(query)
             except AttributeError: 
                print("... predict_proba attribute exists but is not operational (e.g. probability=False).")
                y_proba = None 

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    update_predict_log(country, y_pred, y_proba, target_date_str,
                       runtime, MODEL_VERSION, test=test)
    
    return {'y_pred': y_pred, 'y_proba': y_proba}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict based on the model.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--data_dir", type=str, 
                        default=os.path.join("data","cs-train"), 
                        help="Directory for the data (e.g., data/cs-train).")
    parser.add_argument("--run_predict_test", action="store_true", help="Run the example prediction test in __main__.")
    parser.add_argument("--country", type=str, default='all', help="Country for prediction test.")
    parser.add_argument("--year", type=str, default='2018', help="Year for prediction test.")
    parser.add_argument("--month", type=str, default='01', help="Month for prediction test.")
    parser.add_argument("--day", type=str, default='05', help="Day for prediction test.")

    args = parser.parse_args()

    effective_data_dir = args.data_dir 
    # Ensure effective_data_dir is an existing directory before proceeding
    if not os.path.isdir(effective_data_dir):
        print(f"Error: Provided data directory '{effective_data_dir}' does not exist. Please check the path.")
        # Depending on desired behavior, exit or try a default. For now, print error.
        # sys.exit(1) # Uncomment to exit if data_dir is invalid

    if args.train:
        print("TRAINING MODELS")
        model_train(effective_data_dir, test=True) 

    if args.run_predict_test: 
        print("RUNNING __main__ PREDICTION TEST")
        
        try:
            result = model_predict(args.country, args.year, args.month, args.day, 
                                   data_dir=effective_data_dir, test=True) 
            print(result)
        except Exception as e:
            print(f"Error during __main__ prediction test: {e}")

    # If neither --train nor --run_predict_test is specified, maybe print help or a default message
    if not args.train and not args.run_predict_test:
        print("No action specified. Use --train to train models or --run_predict_test to run a prediction example.")
        parser.print_help()