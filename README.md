# Personalized Learning Recommendation System

## What is included
- `app.py`: Streamlit web application, EDA charts, recommendation ranking, and model evaluation.
- `dataset/students.csv`: 1,000 synthetic student-learning records.
- `dataset/learning_resources.csv`: sample learning resources across five subjects.
- `requirements.txt`: Python dependencies.
- `models/`: trained model is created here after first run.

## Setup on Windows
1. Install Python 3.10 or newer from https://www.python.org/downloads/ and enable **Add Python to PATH**.
2. Extract this folder somewhere easy to find, such as Documents.
3. Open the folder in File Explorer.
4. Click the address bar, type `cmd`, and press Enter.
5. (Recommended) Create a virtual environment:
   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   ```
6. Install packages:
   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start the app:
   ```bat
   streamlit run app.py
   ```
8. A local address such as `http://localhost:8501` will appear. Open it in your browser.
9. In the sidebar, choose a subject and topic, enter the profile details, and click **Generate Recommendations**.

## If `py` is not recognized
Try `python --version`. If that also fails, reinstall Python and select **Add Python to PATH**.

## Notes about the data and model
The supplied data is synthetic and intended for a college project demonstration. Its resource ratings are artificially generated; evaluation metrics are illustrative and must not be reported as evidence that recommendations improve real learning outcomes. The model predicts a synthetic rating and the final list is ranked using topic match, preferred format, difficulty fit, and that prediction. For a real deployment, collect consented and anonymized feedback and evaluate with appropriate ranking metrics or a real user study.

## Useful project explanation
- Preprocessing: missing-value imputation, one-hot encoding, and numerical scaling inside a scikit-learn pipeline.
- Model: Random Forest Regressor, evaluated with a held-out 20% test set.
- Metrics: MAE, RMSE, and R².
- EDA: score histogram, learning-preference bar chart, study-hours scatter plot, subject-average bar chart, correlation heatmap.
