# IBM AI Enterprise Workflow Capstone

This repository contains the files for the IBM AI Enterprise Workflow Capstone project. The project is divided into three parts, each focusing on different aspects of data science and machine learning workflows.

## Part 1: Data Exploration and Hypothesis Testing

### Objectives

1. **Understand the Business Scenario**: Assimilate the business context and articulate testable hypotheses.
2. **Data Requirements**: Define the ideal data needed to address the business opportunity before data ingestion.
3. **Data Ingestion**: Create a Python script to automate data extraction from multiple sources, ensuring error handling and returning a feature matrix.
4. **Exploratory Data Analysis (EDA)**: Investigate relationships between data, targets, and business metrics using EDA tools.
5. **Visualization**: Summarize findings with visualizations.

### Tips

- JSON files may have non-uniform feature names; handle this in your data ingestion function.
- Clean invoice IDs by removing letters for better matching.
- Aggregate transactions by day for time-series preparation.

## Part 2: Time-Series Analysis and Modeling

### Objectives

1. **Model Comparison**: Compare different modeling approaches to address the business opportunity.
2. **Model Iteration**: Modify data transformations, pipeline architectures, and hyperparameters.
3. **Model Deployment Preparation**: Retrain the model on all data using the selected approach.
4. **Summary Report**: Articulate findings in a report.

### Time-Series Analysis

- Use TensorFlow, scikit-learn, and Spark ML for model implementation.
- Engineer features from revenue data for machine learning models.
- Consider recursive and ensemble forecasting for multi-point predictions.

### Tools and Resources

- [statsmodels](https://www.statsmodels.org/dev/tsa.html)
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Prophet](https://research.fb.com/prophet-forecasting-at-scale/)

## Part 3: API Development and Deployment

### Objectives

1. **API Development**: Build a draft API with train, predict, and logfile endpoints.
2. **Containerization**: Use Docker to bundle the API, model, and unit tests.
3. **Test-Driven Development**: Iterate on the API considering scale, load, and drift.
4. **Post-Production Analysis**: Analyze the relationship between model performance and business metrics.

### Deployment

- Prepare the model for deployment with a Flask API for training and prediction.
- Ensure the API accommodates different user needs for prediction dates.
- Use scripts to simulate API queries and monitor performance with time-series plots.

## Additional Resources

- [Gaussian Processes in Time-Series](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2011.0550)
- [Wavelets for Time-Series Forecasts](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142064)

This README provides a comprehensive overview of the project structure and objectives, guiding you through each phase of the capstone project.
