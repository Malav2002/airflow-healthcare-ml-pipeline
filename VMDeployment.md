# Airflow Healthcare ML Pipeline - VM Deployment

Apache Airflow pipeline for predicting cardiovascular disease using Random Forest classification, deployed on Google Cloud Platform virtual machine.

## Project Overview

This project implements an end-to-end machine learning pipeline using Apache Airflow to automate the process of training and evaluating a Random Forest classifier for cardiovascular disease prediction. The pipeline is deployed natively on a GCP VM for production-ready orchestration.

### Key Features

- Automated data loading and preprocessing
- Feature engineering (BMI calculation, blood pressure categorization)
- Class imbalance handling using SMOTE
- Random Forest classification model
- Comprehensive model evaluation (accuracy, precision, recall, F1-score)
- Email notifications with performance metrics
- Production deployment on GCP VM
- Systemd service management for reliability

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

**Size**: 70,000 samples (or 10,000 for sample dataset)

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

### Deployment Architecture

```
GCP VM (e2-standard-2)
├── Apache Airflow 2.7.3
├── Python Virtual Environment
├── SQLite Metadata Database
├── Systemd Services
│   ├── airflow-webserver (port 8080)
│   └── airflow-scheduler
└── Project Files
    ├── DAGs
    ├── Data
    ├── Model Output
    └── Logs
```

## Technology Stack

- **Orchestration**: Apache Airflow 2.7.3
- **Cloud Platform**: Google Cloud Platform (GCP)
- **Compute**: GCP VM Instance (e2-standard-2, 2 vCPUs, 8GB RAM)
- **Machine Learning**: scikit-learn 1.3.2
- **Data Processing**: pandas 2.0.3, numpy 1.24.3
- **Class Balancing**: imbalanced-learn 0.11.0
- **Visualization**: seaborn 0.12.2, matplotlib 3.7.2
- **Database**: SQLite (Airflow metadata)
- **Process Management**: systemd
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
│   └── cardiovascular_data.csv        # Dataset
├── model/                              # Saved models and results
│   ├── cardiovascular_rf_model.pkl    # Trained model
│   ├── scaler.pkl                     # Feature scaler
│   └── evaluation_results.json        # Performance metrics
├── airflow_venv/                       # Python virtual environment
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation

Airflow Home (~airflow/):
├── airflow.cfg                         # Airflow configuration
├── airflow.db                          # SQLite metadata database
├── dags/                               # Symbolic link to project DAGs
├── logs/                               # Airflow execution logs
└── webserver_config.py                 # Web UI configuration
```

## Prerequisites

- Google Cloud Platform account with billing enabled
- Basic understanding of Linux commands and SSH
- Gmail account with app password configured
- GitHub account (optional, for version control)

## Installation and Setup

### Part 1: Create GCP VM Instance

#### Step 1: Create VM

1. Go to Google Cloud Console: https://console.cloud.google.com
2. Navigate to **Compute Engine > VM Instances**
3. Click **Create Instance**
4. Configure:
   - Name: `airflow-healthcare-vm`
   - Region: `us-central1` (or preferred)
   - Zone: `us-central1-c` (or any available)
   - Machine type: `e2-standard-2` (2 vCPUs, 8GB RAM)
   - Boot disk: Ubuntu 22.04 LTS, 30GB
   - Firewall: Allow HTTP and HTTPS traffic
5. Click **Create**

#### Step 2: Configure Firewall

1. Go to **VPC Network > Firewall**
2. Click **Create Firewall Rule**
3. Configure:
   - Name: `allow-airflow-8080`
   - Direction: Ingress
   - Targets: All instances in network
   - Source IP ranges: `0.0.0.0/0`
   - Protocols and ports: `tcp:8080`
4. Click **Create**

#### Step 3: SSH into VM

1. Return to **VM Instances**
2. Click **SSH** button next to your VM
3. Terminal window will open

### Part 2: Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install python3-pip python3-venv python3-full git tmux -y

# Verify installations
python3 --version
pip3 --version
git --version
```

### Part 3: Clone Project Repository

```bash
# Clone project from GitHub
git clone https://github.com/YOUR_USERNAME/airflow-healthcare-ml-pipeline.git

# Navigate to project directory
cd airflow-healthcare-ml-pipeline

# Verify structure
ls -la
```

### Part 4: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv airflow_venv

# Activate virtual environment
source airflow_venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Apache Airflow
pip install apache-airflow==2.7.3

# Install ML dependencies
pip install -r requirements.txt

# Verify installations
python -c "import airflow; print(f'Airflow: {airflow.__version__}')"
python -c "import pandas, sklearn, xgboost; print('All packages installed')"
```

### Part 5: Configure Airflow

#### Step 1: Initialize Airflow Database

```bash
# Set Airflow home directory
export AIRFLOW_HOME=~/airflow

# Fix flask-session compatibility
pip uninstall flask-session -y
pip install 'flask-session<0.6.0'

# Initialize database
airflow db migrate

# Verify database created
ls -la ~/airflow/
```

#### Step 2: Configure airflow.cfg

```bash
nano ~/airflow/airflow.cfg
```

Update these sections:

**DAGs Configuration:**
```ini
[core]
dags_folder = /home/USERNAME/airflow-healthcare-ml-pipeline/dags
enable_xcom_pickling = True
load_examples = False
```

**SMTP Configuration:**
```ini
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your_email@gmail.com
smtp_password = your_gmail_app_password
smtp_port = 587
smtp_mail_from = your_email@gmail.com
```

**API Configuration:**
```ini
[api]
auth_backends = airflow.api.auth.backend.basic_auth,airflow.api.auth.backend.session
```

Replace `USERNAME` with your actual username (check with `whoami`).

Save and exit (Ctrl+X, Y, Enter).

#### Step 3: Create Airflow Admin User

```bash
airflow users create \
  --username admin \
  --firstname Malav \
  --lastname Patel \
  --role Admin \
  --email your_email@gmail.com \
  --password admin123
```

### Part 6: Set Up Project Files

#### Step 1: Create Required Directories

```bash
cd ~/airflow-healthcare-ml-pipeline

# Create directories
mkdir -p data model logs plugins

# Set permissions
chmod 755 dags/ data/ model/
```

#### Step 2: Create Symbolic Link for DAGs

```bash
# Link project DAGs to Airflow
ln -s ~/airflow-healthcare-ml-pipeline/dags ~/airflow/dags

# Verify link
ls -la ~/airflow/dags
```

#### Step 3: Update File Paths in Code

```bash
# Get your username
USERNAME=$(whoami)

# Update paths in model_development.py
sed -i "s|/opt/airflow|/home/$USERNAME/airflow-healthcare-ml-pipeline|g" ~/airflow-healthcare-ml-pipeline/dags/src/model_development.py

# Update paths in success_email.py
sed -i "s|/opt/airflow|/home/$USERNAME/airflow-healthcare-ml-pipeline|g" ~/airflow-healthcare-ml-pipeline/dags/src/success_email.py

# Verify changes
grep -n "airflow-healthcare-ml-pipeline" ~/airflow-healthcare-ml-pipeline/dags/src/model_development.py
```

#### Step 4: Download Dataset

**Option A: Manual Upload via GCP Console**

1. Download from Kaggle: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
2. In GCP Console, go to your VM
3. Click SSH dropdown > Upload file
4. Upload `cardio_train.csv`
5. Move file:

```bash
mv ~/cardio_train.csv ~/airflow-healthcare-ml-pipeline/data/cardiovascular_data.csv
```

**Option B: Create Sample Dataset**

```bash
cd ~/airflow-healthcare-ml-pipeline/data

python3 << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

data = {
    'id': range(n),
    'age': np.random.randint(10000, 25000, n),
    'gender': np.random.choice([1, 2], n),
    'height': np.random.randint(150, 190, n),
    'weight': np.random.randint(50, 120, n),
    'ap_hi': np.random.randint(90, 180, n),
    'ap_lo': np.random.randint(60, 120, n),
    'cholesterol': np.random.choice([1, 2, 3], n),
    'gluc': np.random.choice([1, 2, 3], n),
    'smoke': np.random.choice([0, 1], n),
    'alco': np.random.choice([0, 1], n),
    'active': np.random.choice([0, 1], n),
    'cardio': np.random.choice([0, 1], n)
}

df = pd.DataFrame(data)
df.to_csv('cardiovascular_data.csv', sep=';', index=False)
print(f"Dataset created with {n} rows")
EOF

# Verify
ls -lh cardiovascular_data.csv
head -5 cardiovascular_data.csv
```

### Part 7: Start Airflow Services

#### Option A: Using tmux (Quick Testing)

```bash
# Start tmux session
tmux new -s airflow

# Start webserver
cd ~/airflow-healthcare-ml-pipeline
source airflow_venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow webserver --port 8080

# Split window: Ctrl+B, then Shift+'
# Or: Ctrl+B, then : and type "split-window"

# Start scheduler in new pane
cd ~/airflow-healthcare-ml-pipeline
source airflow_venv/bin/activate
export AIRFLOW_HOME=~/airflow
airflow scheduler

# Detach: Ctrl+B, then D

# Reattach anytime: tmux attach -t airflow
```

## Usage

### Access Airflow Web UI

#### Step 1: Get VM External IP

```bash
curl ifconfig.me
```

Or check in GCP Console under VM Instances.

#### Step 2: Open Browser

Navigate to:
```
http://YOUR_VM_EXTERNAL_IP:8080
```

#### Step 3: Login

- Username: `admin`
- Password: `admin123`

### Trigger the DAG

**Method 1: Via Web UI**

1. Find `healthcare_ml_pipeline` in DAG list
2. Toggle to ON (if paused)
3. Click play button (▶️) to trigger
4. Monitor execution in Graph or Grid view
5. Click on tasks to view logs
