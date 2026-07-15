# Explainable User Behavior Clustering Dashboard

A beginner-friendly full-stack dashboard built with Flask, Vanilla JavaScript, Chart.js, and SHAP.

## Features

- CSV upload and validation
- Data preprocessing (missing values + categorical encoding)
- Engagement score feature engineering
- KMeans clustering + cluster labels
- RandomForest surrogate model + SHAP feature importance
- JSON API backend (no plot images generated)
- Frontend dashboard charts and clustered users table

## Project Structure

```text
project/
├── app.py
├── preprocessing.py
├── clustering.py
├── explainability.py
├── requirements.txt
├── uploads/
│   └── .gitkeep
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Expected CSV Columns

Required columns for analysis:

- `Age`
- `Gender`
- `Membership Type`
- `Total Spent` (or `Total Spend`)
- `Items Purchased`
- `Average Rating`
- `Discount Applied`
- `Days Since Last Purchase`

Optional columns:

- `Customer ID` (dropped during preprocessing)
- `City` (dropped during preprocessing)

## Setup and Run

1. Open terminal in `project/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open browser:
   - [http://127.0.0.1:5000](http://127.0.0.1:5000)

## API Endpoints

- `POST /upload`  
  Upload a CSV file (`multipart/form-data` with key `file`)

- `POST /process`  
  Run preprocessing, clustering, and SHAP explainability

- `GET /results`  
  Returns:
  - cluster distribution
  - scatter points (engagement vs purchases)
  - clustered users table data

- `GET /importance`  
  Returns SHAP feature importance as JSON

## Error Handling Included

- Missing file in upload request
- Invalid extension (non-CSV)
- Empty CSV
- Missing required columns
- Running `/process` before upload
- Running `/results` or `/importance` before processing
