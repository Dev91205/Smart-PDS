# Smart Public Distribution System (SmartPDS)

## 📌 Overview

SmartPDS is an intelligent system designed to improve the efficiency, transparency, and reliability of the Public Distribution System (PDS). It integrates **demand forecasting**, **fraud detection**, and **supply chain optimization** into a unified pipeline.

The project leverages data analytics, machine learning, and linear programming to ensure optimal allocation of resources while minimizing leakage and inefficiencies.

---

## 🚀 Key Features

### 🔹 Demand Forecasting

* Predicts village-level demand using historical consumption data
* Helps prevent shortages and overstocking
* Improves planning accuracy for distribution

### 🔹 Fraud Detection

* Identifies suspicious transactions in ration distribution
* Detects anomalies in FPS (Fair Price Shop) activity
* Enhances system transparency and accountability

### 🔹 Supply Chain Optimization

* Uses Linear Programming (LP) to optimize allocation from warehouses to villages
* Minimizes transportation cost and resource wastage
* Ensures efficient utilization of warehouse stock

---

## 🧠 Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * NumPy
  * Pandas
  * OR-Tools (Linear Optimization)
  * Jupyter Notebook
* **Data Handling:** CSV-based datasets
* **Application Layer:** Python (app.py)

---

## 📂 Project Structure

```
SmartPDS/
│
├── app.py                          # Main application script
├── allocation_plan.csv             # Optimized allocation output
├── warehouse_utilization.csv       # Warehouse usage statistics
├── village_demand_forecast.csv     # Forecasted demand data
│
├── datasets/
│   ├── transactions.csv            # Transaction records
│   ├── ration_cards.csv            # Beneficiary data
│   ├── fps_data.csv                # Fair Price Shop data
│   ├── villages.csv                # Village information
│   ├── fraud_alerts.csv            # Detected fraud cases
│
├── notebooks/
│   ├── SmartPDS_DemandForecasting.ipynb
│   ├── SmartPDS_FraudDetection.ipynb
│   ├── SmartPDS_LP_Optimizer.ipynb
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd SmartPDS
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the Main Application

```bash
python app.py
```

### Run Individual Modules

* Open Jupyter Notebook:

```bash
jupyter notebook
```

* Execute:

  * Demand Forecasting Notebook
  * Fraud Detection Notebook
  * Optimization Notebook

---

## 📊 Outputs

* **Allocation Plan:** Optimized distribution from warehouses to villages
* **Fraud Alerts:** Suspicious transaction records
* **Demand Forecast:** Predicted demand for each village
* **Warehouse Utilization:** Stock usage efficiency

---

## 🎯 Objectives

* Reduce food wastage
* Prevent fraud and leakage
* Improve supply chain efficiency
* Enable data-driven decision-making

---

## ⚠️ Limitations

* Depends heavily on data quality
* Forecasting accuracy may vary with sparse data
* Real-time integration not implemented

---

## 🔮 Future Enhancements

* Real-time data integration (IoT / API-based)
* Advanced ML models (LSTM, XGBoost)
* Dashboard visualization (Streamlit / Power BI)
* GPS-based logistics tracking
* Blockchain for transparency

---

## 🤝 Contributing

Contributions are welcome. Please fork the repository and submit a pull request with detailed changes.

---

## 📜 License

This project is for academic and research purposes.

---

## 👨‍💻 Author

Developed as part of an academic/innovation project to modernize public distribution systems using AI and optimization techniques.
