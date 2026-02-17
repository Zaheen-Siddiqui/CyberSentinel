# 🛡️ CyberSentinel - Network Traffic Classifier

A basic machine learning demonstration project that classifies network traffic as either **Threat** or **Harmless** using the NSL-KDD dataset. This project uses a Random Forest classifier to detect potential intrusions in network traffic patterns.

## 📋 Description

CyberSentinel is a simple intrusion detection system that demonstrates how machine learning can be applied to cybersecurity. The system trains on the NSL-KDD dataset and provides a web interface where users can upload CSV files containing network traffic data to get real-time classification results.

**Note:** This is for demonstration purposes only and should not be used in production environments.

## 📁 Project Structure

```
CyberSentinel/
│
├── data/
│   ├── train/
│   │   └── KDDTrain+.csv          # Training dataset (NSL-KDD)
│   └── test/
│       └── sample_input.csv       # Uploaded CSV files for prediction
│
├── model/
│   └── intrusion_model.pkl        # Trained ML model (generated after training)
│
├── src/
│   ├── train_model.py             # Script to train the model
│   ├── predict.py                 # Script to make predictions
│   └── preprocess.py              # Data preprocessing utilities
│
├── app.py                          # Flask web application
│
├── templates/
│   └── index.html                  # Web interface (single page)
│
├── requirements.txt                # Python dependencies
│
└── README.md                       # Project documentation
```

## 🚀 Quick Start Guide

Get CyberSentinel running in 5 minutes!

### Step 1: Install Dependencies (1 minute)

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### Step 2: Get Training Data (2 minutes)

1. Download NSL-KDD dataset from: https://www.unb.ca/cic/datasets/nsl.html
2. Extract **KDDTrain+.csv**
3. Place it in `data/train/` directory

### Step 3: Train the Model (1 minute)

Run the training script:

```bash
python src/train_model.py
```

Wait for training to complete. You'll see accuracy metrics when done.

### Step 4: Start the Web App (30 seconds)

Launch the Flask application:

```bash
python app.py
```

### Step 5: Use the Classifier (30 seconds)

1. Open browser: http://127.0.0.1:5000
2. Upload a CSV file with network traffic data
3. Click "Classify Traffic"
4. View your results!

## 📋 Detailed Setup and Usage

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Zaheen-Siddiqui/CyberSentinel.git
   cd CyberSentinel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the training data**
   - Download the NSL-KDD dataset (KDDTrain+.csv)
   - Place it in the `data/train/` directory
   - The CSV should contain network traffic features with a label column

### Usage

#### Step 1: Train the Model

Before using the classifier, you need to train the model with the NSL-KDD dataset:

```bash
python src/train_model.py
```

This will:
- Load the training data from `data/train/KDDTrain+.csv`
- Preprocess and clean the data
- Convert labels to binary classification (normal → Harmless, others → Threat)
- Train a Random Forest classifier
- Save the trained model to `model/intrusion_model.pkl`
- Display training accuracy and metrics

#### Step 2: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

#### Step 3: Classify Network Traffic

1. On the web page, click "Choose File" and select a CSV file containing network traffic data
2. Click "Classify Traffic"
3. View the results showing:
   - Total records analyzed
   - Number of threats detected
   - Number of harmless connections
   - Detailed table of predictions

### Command-Line Prediction (Optional)

You can also run predictions from the command line:

```bash
python src/predict.py data/test/sample_input.csv
```

This will output predictions to a new CSV file with "_predictions" suffix.

## 📊 NSL-KDD Dataset Information

### Overview
Each row in the **NSL-KDD dataset** represents **one network connection (session)**. The dataset is used to classify whether the connection is:
- **Normal (Harmless Traffic)**
- **Attack (Malicious Activity)**

### Label/Class Values
The `label` represents the outcome of the connection:
- ✅ **normal**: Legitimate, harmless network traffic
- ❌ **Attack Types**: neptune, smurf, apache2, satan, ipsweep, etc. (representing attack behavior patterns)

### Key Feature Groups

#### 1) Basic Connection Features
- **duration**: Length of connection in seconds
- **protocol_type**: tcp, udp, icmp
- **service**: http, ftp, telnet, smtp, etc.
- **flag**: Connection status (SF, S0, REJ, etc.)
- **src_bytes/dst_bytes**: Bytes transferred

#### 2) Content-Based Features
- **land**: Source and destination IP/port same (suspicious when 1)
- **wrong_fragment**: Incorrect IP fragments count
- **urgent**: Urgent packets count
- **hot**: Sensitive operations count
- **num_failed_logins**: Failed login attempts
- **logged_in**: Login success status
- **num_compromised**: Compromised conditions detected
- **root_shell**: Root shell obtained (strong attack indicator)

#### 3) Time-Based Traffic Features (last 2 seconds)
- **count**: Connections to same host
- **srv_count**: Connections to same service
- **serror_rate**: SYN error percentage
- **rerror_rate**: REJ error percentage
- **same_srv_rate**: Same service connection fraction
- **diff_srv_rate**: Different service connection fraction

#### 4) Host-Based Traffic Features (last 100 connections)
- **dst_host_count**: Connections to same destination host
- **dst_host_srv_count**: Same service connections on host
- Various rate calculations for error detection

### CSV File Requirements

Your input CSV file should:
- Contain the same features as the training data
- Have one row per network connection
- Include numerical and/or categorical features
- **Not require** a label column (the model will predict this)

### Value Ranges & Interpretation
- **Binary fields**: 0 or 1
- **Rate fields**: 0.0 to 1.0
- **Count fields**: Non-negative integers
- **Byte fields**: Non-negative integers
- **Categorical fields**: Must be encoded before ML

### Recommended Preprocessing
- Encode categorical features (`protocol_type`, `service`, `flag`)
- Normalize numeric features
- Drop `difficulty` column
- Convert to binary labels: normal → 0, attack → 1

## 🔍 How It Works

1. **Training Phase:**
   - Loads NSL-KDD training dataset
   - Cleans and preprocesses data
   - Encodes categorical features
   - Converts multi-class labels to binary (Threat/Harmless)
   - Trains Random Forest classifier with 100 trees
   - Saves model and encoders for later use

2. **Prediction Phase:**
   - User uploads a CSV file via web interface
   - System loads the trained model
   - Preprocesses the uploaded data using same encoders
   - Makes predictions for each row
   - Displays results with statistics

## 🛠️ Technologies Used

- **Python** - Programming language
- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **joblib** - Model serialization
- **HTML** - Web interface

## 🔧 Troubleshooting

**Problem:** Model not found error  
**Solution:** Make sure you completed training (Step 3 in Quick Start)

**Problem:** Training data not found  
**Solution:** Ensure KDDTrain+.csv is in `data/train/` directory

**Problem:** CSV upload error  
**Solution:** Verify your CSV has the same features as training data

**Problem:** Categorical encoding errors  
**Solution:** Ensure categorical features (protocol_type, service, flag) have valid values

**Problem:** Prediction accuracy issues  
**Solution:** Check data preprocessing and feature normalization

## ⚠️ Limitations

- This is a basic demonstration project
- Not suitable for production cybersecurity systems
- Model accuracy depends on training data quality
- No user authentication or session management
- Single-page interface only

## 📝 License

This project is for educational purposes only.

## 👤 Author

Zaheen Siddiqui

## 🤝 Contributing

This is a demonstration project and is not actively maintained for contributions.

---

**Disclaimer:** This tool is for educational and demonstration purposes only. Do not use in production environments or rely on it for actual security decisions.
