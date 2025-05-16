
## Usage: Step-by-Step Guide

All commands should be executed from the **root directory** of the project (e.g., `AI-Enterprise-Workflow-Capstone/`).  Ensure your virtual environment is activated before running any commands.

### 1. Training the Model

The `model.py` script handles model training.  This step is necessary before you can generate predictions.

*   **Command:**

    ```bash
    python -m solution_guidance.model --train --data_dir data/cs-train
    ```

    *   `python -m solution_guidance.model`: Executes the `model.py` script as a module.  This is important for Python to correctly resolve relative imports within the `solution_guidance` package.
    *   `--train`:  This flag tells the script to start the model training process.
    *   `--data_dir data/cs-train`:  This specifies the directory where your raw training data is located.

        *   **Important**:  If your data is in a *different* location, replace `data/cs-train` with the correct path. For example:

            ```bash
            python -m solution_guidance.model --train --data_dir /path/to/my/data/
            ```

*   **What Happens:**

    1.  The script will load raw data from the directory you specified with `--data_dir`.  If you don't specify `--data_dir`, it will default to `data/cs-train`.

    2.  It will process this raw data into a time-series format, suitable for the model. As part of this processing, it will create a subdirectory named `ts-data` inside your data directory and save CSV files there.

    3.  It will train one or more machine learning models. The `model.py` script, when run with `--train`, defaults to running in a simplified "test mode" which uses a subset of data and trains models for a limited set of countries to keep training times shorter. These models will be saved with a prefix indicating they are for testing (e.g., `test-all-0_1.joblib`).

        *   To train models using all the data and for all relevant countries, you'll need to adjust the `test` flag directly within the `model.py` script itself (advanced).  This is *not* recommended for initial setup, as it will significantly increase training time.

    4.  The trained model(s) will be saved as `.joblib` files in the `models/` directory.  If the `models/` directory doesn't exist, it will be created.

    5.  The script will also create log files, documenting the training process. These logs are saved in the `logs/` directory.

*   **Example Output (Successful Training):**

    ```
    TRAINING MODELS
    ... test flag on
    ...... subsetting countries
    ...... subsetting countries data
    ... loading ts data from files
    ... Training model for country: all
    ... saving test version of model: models\test-all-0_1.joblib
    ... Training model for country: united_kingdom
    ... saving test version of model: models\test-united_kingdom-0_1.joblib
    ```

### 2. Making Predictions

Once you have trained the model, you can use it to make predictions. The `model.py` script includes an example prediction routine that can be run from the command line.

*   **Command:**

    ```bash
    python -m solution_guidance.model --run_predict_test --data_dir data/cs-train --country all --year 2018 --month 01 --day 05
    ```

    *   `python -m solution_guidance.model`:  Executes the `model.py` script as a module.
    *   `--run_predict_test`:  This flag tells the script to run the example prediction test.
    *   `--data_dir data/cs-train`:  This is *required* and tells the script where to find the processed data that the model needs for prediction. If you used a different `--data_dir` during training, make sure you use the *same* path here.
    *   `--country all`:  This specifies the country for which to make the prediction.  Possible values depend on your training data and the models that were successfully trained.  The training output will show what models were trained (e.g.,  `test-all-0_1.joblib`, `test-united_kingdom-0_1.joblib`).
    *   `--year 2018 --month 01 --day 05`:  These specify the date for which the prediction should be made (year, month, and day).

*   **What Happens:**

    1.  The script will load the appropriate model (e.g., `test-all-0_1.joblib`) from the `models/` directory. It will also load the processed data from your specified `--data_dir`.  It will prioritize loading test models (files starting with "test-").

    2.  It will extract the data for the specified country and date.

    3.  It will feed this data to the trained model to generate a prediction.

    4.  The prediction results (predicted values and, optionally, probability scores, if available from the model) will be printed to the console.

    5.  A record of the prediction will be written to the `logs/predict_log.json` file.

*   **Example Output:**

    ```
    RUNNING __main__ PREDICTION TEST
    ... Target date: 2018-01-05
    {'y_pred': array([1234.567]), 'y_proba': None}
    ```
    The  `'y_pred'`  value (1234.567 in this example) is the predicted revenue. The  `'y_proba'`  value will be  `None`  for regression models like the Random Forest Regressor used in this project.

### 3. Running Automated Tests

To ensure the code is working correctly, you can run the automated tests included in the `tests/` directory.

*   **Command:**

    ```bash
    python -m pytest tests/ -v
    ```

    *   `python -m pytest`:  Runs the `pytest` testing framework.
    *   `tests/`:  Specifies the directory where the test files are located.
    *   `-v`:  Enables verbose output, showing details of each test.

*   **What Happens:**

    1.  `pytest`  will search for and execute all files in the  `tests/`  directory that begin with  `test_`.

    2.  It will run all test functions within those files.

    3.  It will report the outcome of each test:  `PASSED`,  `FAILED`, or  `SKIPPED`.

*   **Example Output (Successful Tests):**

    ```
    ============================= test session starts ==============================
    ...
    tests/test_model.py::test_model_accuracy PASSED
    ...
    ============================== 1 passed in 1.23s ===============================
    ```

    If any tests fail, carefully examine the output to determine the cause. Common issues include incorrect file paths, missing data, or problems with the model or data processing logic.

## Advanced Usage

### Training Production Models

The steps above train models in "test mode", suitable for development and experimentation. To train models for actual deployment, you would typically want to train on the entire dataset and potentially train models for all available countries. Here's how (note this is an *advanced* topic):

1.  **Modify `model.py`**: Find the `_model_train` function and the `model_train` function. Look for where the `test` flag is being used.
    *   The `_model_train` function contains the logic that actually subsets data.
    *   The `model_train` function iterates through the countries and calls `_model_train`, and might have conditional checks that skip countries if `test=True`.
2.  **Remove or Comment Out `test` related conditional logic**: The goal is that *all* the available data is used, and models are trained for *all* the countries.
3.  **Run Training**: Execute the training command as before:
    ```bash
    python -m solution_guidance.model --train --data_dir data/cs-train
    ```
    *   Since you modified `model.py` and have removed the test subsetting conditional, now full production models will be created. This is a more time-consuming process. The models will now have the production filename style (e.g., `sl-all-0_1.joblib`, `sl-united_kingdom-0_1.joblib`)

*   **Note:** This requires modifying code, so only attempt this if you have sufficient understanding of Python and the project structure. Always back up the original file.

### Configuring the API

The project includes a FastAPI-based API that allows you to interact with trained models through HTTP requests. This aspect of the project relies on the `app.py` script and would be relevant for production deployments. Details on this are omitted here for brevity (as this novice guide focused on CLI interaction).

## Troubleshooting

*   **Common Beginner Mistakes:**

    *   **Forgetting to activate the virtual environment:** Always ensure your virtual environment is active (`(.venv)` in your terminal prompt) before installing packages or running scripts.
    *   **Running commands from the wrong directory:** All commands should be executed from the `AI-Enterprise-Workflow-Capstone/` root directory of the project.
    *   **Incorrect `--data_dir` path:** Double-check the path you provide with `--data_dir`. It *must* be a valid path to your data directory *relative* to your current location (which should be the project root).
    *   **Mixing Test and Production Models:** If you trained with test data, make sure to load the corresponding test models. Similarly, load production models for production usage. Model filenames reflect the "test" or "sl" (standard/production) configuration.

*   **`ModuleNotFoundError: No module named 'xxx'`**:  This generally indicates a problem with your Python environment. Ensure you have:

    *   Activated your virtual environment.
    *   Installed all dependencies listed in `requirements.txt` (using  `pip install -r requirements.txt`).
    *   Used the  `python -m`  syntax when running scripts that are part of a package (e.g.,  `python -m solution_guidance.model`).  This ensures Python can find the script within the  `solution_guidance`  package.

*   **`FileNotFoundError` for data files:** Make sure the files exist in directory you specified in `data_dir`.

*   **KeyError or IndexError**: This often occurs if there's an error in data loading / data transformation. Ensure you didn't modify any data transformation that might broke the code.

## Contributing

We welcome contributions to this project! Here's how you can help:

1. **Reporting Issues**:
   - Open a GitHub issue to report bugs or suggest improvements
   - Include clear steps to reproduce any problems

2. **Making Changes**:
   - Fork the repository and create a feature branch
   - Commit your changes with descriptive messages
   - Open a pull request against the main branch

3. **Development Guidelines**:
   - Ensure your code passes all existing tests
   - Update documentation when making changes
   - Follow the existing code style
   - Write unit tests for new functionality

4. **Code Review**:
   - All pull requests will be reviewed
   - Be prepared to discuss and iterate on your changes


## License

This project is licensed under the MIT License.