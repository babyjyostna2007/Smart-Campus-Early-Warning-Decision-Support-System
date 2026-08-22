# 🏫 Smart Campus: Early-Warning and Decision-Making System

## 📌 Project Overview

The **Smart Campus Early-Warning and Decision-Making System** is a Data Science and Machine Learning project designed to monitor student, campus, and environmental conditions and identify potential risks before they become serious problems.

The system predicts risk levels, identifies important risk factors, and provides recommended actions through an interactive Streamlit dashboard.

## 🎯 Objectives

- Analyze student academic performance and attendance.
- Monitor campus conditions and resource usage.
- Monitor environmental conditions.
- Detect potential risks at an early stage.
- Predict **Low, Medium, and High** risk levels.
- Identify major risk factors.
- Provide recommended actions to management.
- Present insights through an interactive dashboard.

## 📊 Dataset

### 👨‍🎓 Student-Related
- Student ID
- Attendance
- Current GPA
- Previous GPA
- Assignment Rate
- Backlogs

### 🏫 Campus-Related
- Occupancy Rate
- Electricity Usage
- Internet Usage
- Maintenance Complaints

### 🌦️ Environmental
- Temperature
- Humidity
- Rainfall
- Water Level
- Air Quality Index

### ⚠️ Decision Data
- Risk Score
- Risk Level
- Risk Factors
- Recommended Action

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Train-Test Split
     ↓
Random Forest Classifier
     ↓
Risk Prediction
     ↓
Risk Factor Identification
     ↓
Recommended Action
     ↓
Streamlit Dashboard
```

## 🧹 Data Preprocessing

The raw dataset is cleaned before Machine Learning.

Steps include:
- Checking missing values
- Removing duplicate records
- Filling missing numerical values using the median
- Handling missing categorical values
- Correcting invalid percentage values
- Correcting invalid negative values
- Detecting potential outliers using the IQR method

Output file:

`cleaned_smart_campus.csv`

## 🤖 Machine Learning

The project uses a **Random Forest Classifier** to predict risk level.

### Input Features

```text
Attendance
Current_GPA
Previous_GPA
Assignment_Rate
Backlogs
Occupancy_Rate
Electricity_Usage
Internet_Usage
Maintenance_Complaints
Temperature
Humidity
Rainfall
Water_Level
Air_Quality_Index
```

### Target

`Risk_Level`

Possible predictions:
- Low
- Medium
- High

## 📈 Model Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Feature importance is also calculated to understand which factors influence predictions.

## 💾 Trained Model

The trained Random Forest model is saved as:

`smart_campus_model.pkl`

This allows the Streamlit application to load the trained model without retraining it every time.

## 📊 Dashboard

The Streamlit dashboard contains six tabs:

1. **🏠 Dashboard** – Overall Smart Campus overview.
2. **👨‍🎓 Student-Related** – Academic information and student risk.
3. **🏫 Campus-Related** – Campus usage and maintenance information.
4. **🌦️ Environment-Related** – Environmental conditions.
5. **📊 Charts** – Risk distribution, GPA/attendance analysis, rainfall/water-level analysis, and correlation heatmap.
6. **✅ Conclusion** – Findings, benefits, and future scope.

## 🚨 Example Output

```text
Student ID       : ST1025

Risk Score       : 84%
Risk Level       : HIGH 🔴

Risk Factors:
• Low attendance
• Declining GPA
• Low assignment completion

Recommended Action:
• Academic counselling
• Faculty intervention
• Monitor attendance
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data preprocessing and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Random Forest | Risk classification |
| Joblib | Saving/loading ML model |
| Streamlit | Interactive dashboard |
| CSV | Dataset storage |

## 📁 Project Structure

```text
smart-campus-early-warning-system/
│
├── app.py
├── Smart_Campus_Final_Dataset.csv
├── cleaned_smart_campus.csv
├── smart_campus_model.pkl
├── preprocessing.py
├── train_model.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-campus-early-warning-system.git
```

### 2. Open the project folder

```bash
cd smart-campus-early-warning-system
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

## 💡 Benefits

- Early identification of potential risks.
- Data-driven decision-making.
- Helps management prioritize interventions.
- Combines student, campus, and environmental information.
- Provides visual insights through charts.
- Can be extended to real-time campus monitoring.

## 🚀 Future Scope

- Connect to real-time college databases.
- Integrate IoT sensors.
- Add real-time attendance monitoring.
- Add live environmental monitoring.
- Send SMS/email alerts.
- Deploy across multiple campuses.
- Use continuously updated real-world data.
- Add advanced Machine Learning models.

## ⚠️ Note

This project is a **prototype for educational and demonstration purposes**. The dataset should be replaced with properly collected and validated real-world campus data before using the system for actual institutional decisions.
