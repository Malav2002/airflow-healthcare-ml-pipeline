# Airflow Healthcare ML Pipeline

Apache Airflow pipeline for predicting cardiovascular disease using Random Forest classification with comprehensive model evaluation metrics.

## Project Overview

This project implements an end-to-end machine learning pipeline using Apache Airflow to automate the process of training and evaluating a Random Forest classifier for cardiovascular disease prediction. The pipeline is containerized using Docker Compose for easy deployment and reproducibility.

### Key Features

- Automated data loading and preprocessing
- Feature engineering (BMI calculation, blood pressure categorization)
- Class imbalance handling using SMOTE
- Random Forest classification model
- Comprehensive model evaluation (accuracy, precision, recall, F1-score)
- Email notifications with performance metrics
- Dockerized deployment for portability

## Dataset

**Source**: Cardiovascular Disease Dataset from Kaggle

**Features**:
- Age (converted from days to years)
- Gender
- Height and Weight
- Blood Pressure (systolic/diastolic)
- Cholesterol levels
- Glucose levels
- Smoking status
- Alcohol intake
- Physical activity
- BMI (calculated feature)
- Blood pressure category (derived feature)

**Target**: Binary classification for presence of cardiovascular disease

**Size**: Approximately 70,000 samples after preprocessing

## Architecture

### DAG Workflow

```
load_data 
    ↓
preprocess_data 
    ↓
separate_data 
    ↓
train_model 
    ↓
evaluate_model 
    ↓
send_success_email
```

### Task Descriptions

1. **load_data**: Loads the cardiovascular disease dataset from CSV
2. **preprocess_data**: Cleans data, removes outliers, engineers features, scales data
3. **separate_data**: Extracts data components for downstream tasks
4. **train_model**: Trains Random Forest classifier and saves model to disk
5. **evaluate_model**: Evaluates model performance and generates metrics
6. **send_success_email**: Sends HTML email with results and metrics

## Technology Stack

- **Orchestration**: Apache Airflow 2.7.3
- **Containerization**: Docker & Docker Compose
- **Machine Learning**: scikit-learn 1.3.2, XGBoost 2.0.3
- **Data Processing**: pandas 2.0.3, numpy 1.24.3
- **Class Balancing**: imbalanced-learn 0.11.0
- **Visualization**: seaborn 0.12.2, matplotlib 3.7.2
- **Database**: PostgreSQL 13
- **Python**: 3.10

## Project Structure

```
airflow-healthcare-ml-pipeline/
├── dags/
│   ├── healthcare_dag.py              # Main DAG definition
│   └── src/
│       ├── model_development.py       # ML pipeline functions
│       └── success_email.py           # Email notification handler
├── data/
│   └── cardiovascular_data.csv        # Dataset (not tracked in git)
├── model/                              # Saved models and results
│   ├── cardiovascular_rf_model.pkl    # Trained model
│   ├── scaler.pkl                     # Feature scaler
│   └── evaluation_results.json        # Performance metrics
├── logs/                               # Airflow logs
├── plugins/                            # Custom Airflow plugins
├── docker-compose.yaml                 # Docker services configuration
├── Dockerfile                          # Custom Airflow image
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (not tracked)
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation
```

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0+
- At least 4GB RAM available for Docker
- 5GB free disk space
- Gmail account with app password (for email notifications)

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/airflow-healthcare-ml-pipeline.git
cd airflow-healthcare-ml-pipeline
```

### Step 2: Download Dataset

Download the Cardiovascular Disease dataset from Kaggle:
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Place the downloaded CSV file in the `data/` directory and rename it to `cardiovascular_data.csv`.

Alternatively, create sample data for testing:

```bash
cd data
python3 create_sample_data.py
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Airflow User ID (use 50000 for Mac/Windows, $(id -u) for Linux)
AIRFLOW_UID=50000

# Airflow Admin Credentials
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123

# SMTP Configuration for Gmail
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
```

**Note**: Never commit the `.env` file to git. It is already included in `.gitignore`.

### Step 4: Update Email Recipient

Edit `dags/src/success_email.py` and update line 28 with your email address:

```python
receiver_email = 'your_email@northeastern.edu'
```

### Step 5: Build and Start Airflow

```bash
# Stop any existing containers
docker compose down -v

# Build custom Airflow image with ML packages
docker compose build

# Initialize Airflow database and create admin user
docker compose up airflow-init

# Start all services in detached mode
docker compose up -d
```

### Step 6: Verify Installation

```bash
# Check container status (all should be healthy)
docker compose ps

# Verify Python packages are installed
docker compose exec airflow-scheduler python -c "import pandas, sklearn, xgboost, imblearn; print('All packages installed')"

# Check if DAG is loaded
docker compose exec airflow-scheduler airflow dags list | grep healthcare
```

## Usage

### Access Airflow Web UI

Open your browser and navigate to: http://localhost:8080

**Login Credentials**:
- Username: admin
- Password: admin123

### Trigger the DAG

**Method 1: Web UI**
1. Find `healthcare_ml_pipeline` in the DAG list
2. Toggle the DAG to ON (if paused)
3. Click the play button to trigger the DAG
4. Monitor execution in the Graph or Tree view

**Method 2: Command Line**
```bash
# Trigger DAG manually
docker compose exec airflow-scheduler airflow dags trigger healthcare_ml_pipeline

# Monitor execution
docker compose logs -f airflow-scheduler

# Check DAG run status
docker compose exec airflow-scheduler airflow dags list-runs -d healthcare_ml_pipeline
```

### View Task Logs

**Via Web UI**:
1. Click on the DAG name
2. Select Graph view
3. Click on any task circle
4. Click "Log" button

**Via Command Line**:
```bash
# View specific task logs
docker compose exec airflow-scheduler airflow tasks logs healthcare_ml_pipeline load_data latest
docker compose exec airflow-scheduler airflow tasks logs healthcare_ml_pipeline evaluate_model latest
```

### Check Results

After successful execution:

```bash
# List generated model files
ls -lh model/

# View evaluation metrics
cat model/evaluation_results.json

# Format JSON output
python3 -c "import json; print(json.dumps(json.load(open('model/evaluation_results.json')), indent=2))"
```

## Model Performance

The Random Forest classifier achieves the following performance metrics on the test set:

- **Accuracy**: ~72-74%
- **Precision**: ~71-73%
- **Recall**: ~73-75%
- **F1-Score**: ~72-74%

### Top Important Features

1. Systolic Blood Pressure (ap_hi)
2. Age
3. Weight
4. BMI (Body Mass Index)
5. Cholesterol Level

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)
```

## Modifications from Base Lab

This project includes the following enhancements beyond the original lab requirements:

1. **Dataset Change**: Switched from advertising dataset to cardiovascular disease dataset
2. **Advanced Model**: Upgraded from Logistic Regression to Random Forest
3. **Comprehensive Metrics**: Added precision, recall, F1-score, and confusion matrix
4. **Feature Engineering**: Implemented BMI calculation and blood pressure categorization
5. **Class Balancing**: Applied SMOTE to handle class imbalance
6. **Outlier Removal**: Added data quality checks and outlier filtering
7. **Docker Deployment**: Containerized using Docker Compose for reproducibility
8. **Custom Image**: Built custom Airflow image with pre-installed ML packages
9. **Enhanced Logging**: Detailed logging at each pipeline stage
10. **HTML Email**: Formatted email notifications with metrics table

## Troubleshooting

### Issue: Containers won't start

```bash
# Check logs
docker compose logs postgres
docker compose logs airflow-scheduler

# Restart services
docker compose restart
```

### Issue: DAG shows as broken

```bash
# Check for Python import errors
docker compose exec airflow-scheduler python /opt/airflow/dags/healthcare_dag.py

# Verify packages are installed
docker compose exec airflow-scheduler pip list | grep scikit-learn

# Restart scheduler
docker compose restart airflow-scheduler
```

### Issue: Dataset not found

```bash
# Verify file location
ls -lh data/cardiovascular_data.csv

# Check file is accessible in container
docker compose exec airflow-scheduler ls -lh /opt/airflow/data/
```

### Issue: Email not sending

Check SMTP credentials in `.env` file and ensure Gmail app password is correct. Email failure does not affect the ML pipeline execution.

### Issue: Out of memory

Reduce dataset size by sampling in `model_development.py`:

```python
data = data.sample(n=10000, random_state=42)
```

## Maintenance

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver
```

### Stop Services

```bash
# Stop containers
docker compose stop

# Stop and remove containers
docker compose down

# Stop and remove containers + volumes (fresh start)
docker compose down -v
```

### Rebuild After Changes

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Clear Failed DAG Runs

```bash
# Clear specific task
docker compose exec airflow-scheduler airflow tasks clear healthcare_ml_pipeline -t task_name

# Clear entire DAG run
docker compose exec airflow-scheduler airflow dags delete healthcare_ml_pipeline
```

## Development

### Adding New Python Packages

1. Add package to `requirements.txt`
2. Rebuild Docker image:
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Modifying the DAG

1. Edit files in `dags/` directory
2. Changes are automatically detected (volume mounted)
3. Wait for scheduler to reload (30-60 seconds)
4. Refresh Airflow UI

### Testing Changes Locally

```bash
# Test Python script directly
docker compose exec airflow-scheduler python /opt/airflow/dags/src/model_development.py

# Test DAG parsing
docker compose exec airflow-scheduler python /opt/airflow/dags/healthcare_dag.py

# List all DAGs
docker compose exec airflow-scheduler airflow dags list
```

## Security Considerations

- `.env` file contains sensitive credentials and is excluded from git
- SMTP password uses Gmail app password, not account password
- Containers run with non-root user for security
- Database credentials should be changed for production use

## Performance Optimization

- LocalExecutor for single-node deployment
- XCom pickling enabled for efficient data passing
- Model files saved to persistent volume
- PostgreSQL for metadata storage
- Resource limits can be added to docker-compose.yaml if needed

## Future Enhancements

Potential improvements for this project:

- Add XGBoost model comparison
- Implement hyperparameter tuning with GridSearchCV
- Create visualization dashboard with model metrics
- Add model versioning and experiment tracking
- Implement A/B testing for model comparison
- Add data quality checks and validation
- Create CI/CD pipeline for automated testing
- Add model monitoring and drift detection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Author

**Malav Patel**  
Master's in Computer Science  
Northeastern University  
Email: patel.malav1@northeastern.edu

## Acknowledgments

- Apache Airflow documentation and community
- Kaggle for the cardiovascular disease dataset
- Northeastern University MLOps course materials
- scikit-learn and XGBoost documentation

## References

- Apache Airflow: https://airflow.apache.org/
- Docker Documentation: https://docs.docker.com/
- scikit-learn: https://scikit-learn.org/
- Cardiovascular Disease Dataset: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: patel.malav1@northeastern.edu

---

Last Updated: February 2026