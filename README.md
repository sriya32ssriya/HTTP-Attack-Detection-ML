# XSS and SQL Injection Detection using Machine Learning

This project is focused on detecting **malicious HTTP requests**, specifically **XSS (Cross-Site Scripting)** and **SQL injection attacks**, using machine learning. The goal is to automatically classify requests as **safe** or **malicious**, helping improve web security.

---

## Project Features

- Detect XSS and SQL injection attacks in HTTP requests  
- Preprocess and analyze HTTP request text for features  
- Compare multiple ML models: Logistic Regression, SVM, and Naive Bayes  
- Selected Multinomial Naive Bayes for highest accuracy  
- Includes Python scripts for training, predicting, and evaluating models  
- Jupyter notebook provides step-by-step analysis  
- PDF report explains methodology, dataset, and results  

---

## Tools & Technologies

- **Python** – main programming language  
- **Scikit-learn** – ML algorithms and evaluation metrics  
- **Pandas & NumPy** – data preprocessing and analysis  
- **Jupyter Notebook** – interactive data exploration and visualization  
- **Git & GitHub** – version control  

---

## Project Structure


XSS-SQLi-Detection-ML/
│
├── dataset/ # HTTP request data from Kaggle and GitHub
├── train.py # Script to train the ML model
├── predict.py # Script to classify new HTTP requests
├── model.py # Implementation of ML models
├── notebook.ipynb # Analysis and evaluation in Jupyter
├── XSS-SQLi_Report.pdf # PDF report of project methodology and results
├── README.md # Project documentation
└── .gitignore # Ignores unnecessary files


---

## Dataset

- Contains labeled HTTP requests as **normal** or **malicious**  
- Data sourced from **Kaggle** and **GitHub**  
- Cleaned and transformed into features suitable for machine learning  

---

## Methodology

1. **Data Preprocessing:**  
   - Cleaned HTTP request strings  
   - Tokenized request parameters and values  
   - Converted text into numerical features for ML models  

2. **Model Evaluation:**  
   - Tested **Logistic Regression**, **SVM**, and **Multinomial Naive Bayes**  
   - Compared performance metrics (accuracy, precision, recall)  
   - Chose **Multinomial Naive Bayes** as it performed best  

3. **Training & Prediction:**  
   - Trained the selected model on labeled dataset  
   - Predicted new HTTP requests as safe or malicious  
   - Evaluated model performance using standard metrics  

---

## Results

- Naive Bayes achieved the **highest accuracy** among all tested models  
- Detailed evaluation, confusion matrix, and analysis are included in `notebook.ipynb`  
- PDF report (`XSS-SQLi_Report.pdf`) explains methodology, dataset, and results in detail  

---

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/XSS-SQLi-Detection-ML.git

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Predict new HTTP requests:

python predict.py

Open notebook.ipynb to view analysis and results

Future Work

Implement real-time detection for HTTP traffic

Try other ML models (Random Forest, SVM with fine-tuning)

Expand dataset with more diverse attack patterns

Build a web interface for easier interaction

Purpose

The goal of this project is to apply machine learning to improve web security by automatically detecting malicious HTTP requests. By comparing different models and selecting the most accurate one, this project demonstrates a practical approach to cybersecurity using ML.
