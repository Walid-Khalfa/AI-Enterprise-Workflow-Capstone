import logging
import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from solution_guidance.model import model_train, model_predict, MODEL_DIR
import time

app = FastAPI()

# Configure logging
LOG_FILE = 'api.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictionInput(BaseModel):
    country: str
    year: str
    month: str
    day: str

@app.on_event("startup")
async def startup_event():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    logging.info("Application startup: Model directory ensured.")

@app.post("/train")
def train_model_endpoint():
    """
    Endpoint to trigger model training.
    Assumes training data is in 'cs-train' directory relative to the project root.
    """
    try:
        logging.info("Training process initiated.")
        # Assuming 'cs-train' is in the parent directory of 'solution_guidance'
        # Adjust path if Dockerfile WORKDIR or project structure is different
        data_dir = os.path.join("..", "cs-train") 
        if not os.path.exists(data_dir):
            # Fallback for Docker context where WORKDIR is /app
            data_dir_docker = "cs-train" # if cs-train is copied to /app/cs-train
            if os.path.exists(data_dir_docker):
                data_dir = data_dir_docker
            else:
                logging.error(f"Training data directory not found at {data_dir} or {data_dir_docker}")
                raise HTTPException(status_code=500, detail=f"Training data directory not found.")

        model_train(data_dir=data_dir, test=False) # Use test=False for actual training
        logging.info("Training process completed successfully.")
        return {"status": "Training completed successfully"}
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post("/predict")
def make_prediction_endpoint(data: PredictionInput):
    """
    Endpoint to make predictions.
    Expects country, year, month, day in the request body.
    """
    try:
        logging.info(f"Prediction initiated for {data.country} on {data.year}-{data.month}-{data.day}.")
        # model_predict expects all_models to be loaded or loads them itself.
        # It also expects data_dir to be set correctly if models are not pre-loaded.
        # The model_load function within model_predict defaults to "../data/cs-train"
        # This might need adjustment based on Docker WORKDIR and where data is mounted/copied.
        # For now, we rely on model_predict's internal model loading.
        prediction_result = model_predict(country=data.country, 
                                          year=data.year, 
                                          month=data.month, 
                                          day=data.day,
                                          test=False) # Use test=False for actual predictions
        logging.info(f"Prediction successful: {prediction_result}")
        return {"prediction": prediction_result.get('y_pred').tolist() if prediction_result.get('y_pred') is not None else [], 
                "probability": prediction_result.get('y_proba').tolist() if prediction_result.get('y_proba') is not None else None}
    except FileNotFoundError as e:
        logging.error(f"Model file not found during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=404, detail=f"Model not found. Please train the model first. Error: {str(e)}")
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/logs")
def get_logs_endpoint():
    """
    Endpoint to retrieve API logs.
    """
    try:
        logging.info("Log retrieval requested.")
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = f.readlines()
            return {"logs": [line.strip() for line in logs[-100:]]} # Return last 100 lines
        else:
            logging.warning("Log file not found.")
            return {"logs": [], "message": "Log file not found."}
    except Exception as e:
        logging.error(f"Error retrieving logs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

# Middleware de logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logging.info(f"Request received: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"Request processed: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s")
    return response