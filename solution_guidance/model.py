import time
import os
import re
import joblib
import numpy as np
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
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices) # np.in1d is deprecated, consider np.isin
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a random forest model
    # Updated criterion values for scikit-learn compatibility
    param_grid_rf = {
    'rf__criterion': ['squared_error','absolute_error'], # Changed 'mse' to 'squared_error', 'mae' to 'absolute_error'
    'rf__n_estimators': [10,15,20,25]
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor(random_state=42))]) # Added random_state for reproducibility
    
    # Removed iid=False as it's deprecated/removed and default behavior matches old iid=False
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1) 
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub(r"\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(grid,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=test) # Changed test=True to test=test
  

def model_train(data_dir,test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subsetting countries") 
        print("...... subsetting countries data") 
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df_country in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df_country,country,test=test)
    
def model_load(prefix='sl',data_dir=None,training=True):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("..","data","cs-train")
    
    models_path = MODEL_DIR 
    if not os.path.isdir(models_path):
         # if models directory does not exist, maybe models were never trained
        if prefix == 'sl': # only raise if default models are expected
            raise Exception(f"Models directory '{models_path}' not found. Did you train the models?")
        else: # if looking for specific prefix and dir not there, return empty
            return {}, {}


    models = [f for f in os.listdir(models_path) if f.startswith(prefix+"-") and f.endswith(".joblib")]


    if not models and prefix == 'sl': # only raise if default models are expected and not found
        raise Exception("Models with prefix '{}' cannot be found in '{}'. Did you train?".format(prefix, models_path))

    all_models = {}
    for model_file in models: 
        try:
            # Assuming format like "prefix-country-version.joblib" or "prefix-country.joblib"
            # More robust parsing: remove prefix, suffix, then split by '-'
            name_part = model_file[len(prefix)+1:-len(".joblib")] #  e.g., "all-0_1" or "all"
            country_key = name_part.split('-')[0] # take the first part as country
            all_models[country_key] = joblib.load(os.path.join(models_path,model_file))
        except Exception as e: # Broad exception to catch parsing/loading errors
            print(f"Warning: Could not parse or load model file name '{model_file}'. Error: {e}. Skipping.")
            continue

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df_country in ts_data.items(): 
        X,y,dates_pd = engineer_features(df_country,training=training) 
        dates_str = np.array([str(d.date()) for d in dates_pd]) # Ensure only date part, as string
        all_data[country] = {"X":X,"y":y,"dates": dates_str}
        
    return(all_data, all_models)


def model_predict(country,year,month,day,all_data=None, all_models=None,test=False, data_dir=None):
    """
    Example function to predict from model.
    'all_data' and 'all_models' can be passed to avoid reloading.
    'data_dir' is used if 'all_data' or 'all_models' need to be loaded.
    """

    time_start = time.time()

    if all_data is None or all_models is None:
        print("... Loading data and models for prediction.")
        # Determine data_dir for loading if not provided
        current_data_dir = data_dir if data_dir else os.path.join("..","data","cs-train")
        # When predicting, features should be engineered as for test/unseen data
        all_data, all_models = model_load(data_dir=current_data_dir, training=False) 

    if country not in all_models:
        raise Exception(f"ERROR (model_predict) - model for country '{country}' could not be found.")
    if country not in all_data:
        raise Exception(f"ERROR (model_predict) - data for country '{country}' could not be found.")

    for d_val in [year, month, day]:
        if not isinstance(d_val, str) or re.search(r"\D", d_val): # ensure d_val is string for re.search
            raise Exception("ERROR (model_predict) - invalid year, month, or day (must be strings of digits).")
    
    model = all_models[country]
    data_country = all_data[country] # Renamed to avoid conflict

    target_date_str = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    print(f"... Target date: {target_date_str}")

    if target_date_str not in data_country['dates']:
        raise Exception(
            f"ERROR (model_predict) - date {target_date_str} not in range "
            f"{data_country['dates'][0]} - {data_country['dates'][-1]}"
        )
    
    date_indx = np.where(data_country['dates'] == target_date_str)[0][0]
    query = data_country['X'].iloc[[date_indx]]
    
    if data_country['dates'].shape[0] != data_country['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch between dates and X.")

    y_pred = model.predict(query)
    y_proba = None
    if hasattr(model, 'predict_proba'): # Simpler check
        # For RandomForestRegressor, predict_proba is not standard.
        # This block might be more relevant for classification models.
        # Checking if it's a classifier or if 'probability=True' was set (for some specific estimators)
        if callable(model.predict_proba): # Check if it's actually callable
             try:
                y_proba = model.predict_proba(query)
             except AttributeError: # Some models have predict_proba but it's conditional (e.g. SVC with probability=False)
                print("... predict_proba attribute exists but is not operational (e.g. probability=False).")
                y_proba = None # Ensure y_proba is None

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    update_predict_log(country, y_pred, y_proba, target_date_str,
                       runtime, MODEL_VERSION, test=test)
    
    return {'y_pred': y_pred, 'y_proba': y_proba}


if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    train_data_dir = os.path.join("..","data","cs-train") 
    model_train(train_data_dir,test=True)

    ## load the model
    print("LOADING MODELS")
    # For prediction, data should be loaded with training=False
    # The model_load function is called inside model_predict if all_data/all_models are not provided.
    # We can pre-load them here if desired for repeated predictions.
    # loaded_all_data, loaded_all_models = model_load(data_dir=train_data_dir, training=False)
    # print("... models loaded: ",",".join(loaded_all_models.keys()))

    ## test predict
    country_predict='all'
    year_predict='2018'
    month_predict='01'
    day_predict='05'
    
    # Call model_predict allowing it to load its own data/models
    # This ensures it uses training=False for loading data via its internal model_load call.
    result = model_predict(country_predict,year_predict,month_predict,day_predict, 
                           data_dir=train_data_dir, test=True)
    print(result)