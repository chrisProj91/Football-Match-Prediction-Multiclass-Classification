Premier League Match Outcome: Multiclass Probabilistic Classification
This project implements a multiclass probabilistic classification model to predict the outcomes of English Premier League matches (Home Win, Draw, Away Win). Instead of simple labels, the model focuses on generating calibrated probabilities, which are crucial for assessing risk and value in sports analytics.

📌 Project Overview
The goal is to predict the probability of the three possible outcomes (H, D, A) using historical match data while strictly avoiding data leakage. The project employs a rolling season-by-season training strategy to simulate real-world conditions where the model is updated as new data becomes available.

🛠️ Tech Stack
Language: Python

Data Manipulation: pandas, numpy

Machine Learning: scikit-learn (Logistic Regression, StandardScaler)

Metrics & Evaluation: Log Loss, Brier Score, Calibration Curves

Visualization: matplotlib

📊 Data & Feature Engineering
The dataset consists of Premier League matches from seasons 2018/19 to 2023/24, sourced from football-data.co.uk.

Key Features Engineered:
To ensure the model only uses information available before kickoff, I developed the following features:

Form Metrics: 5-game rolling average of Goals For (GF) and Goals Against (GA) for both Home and Away teams.

Relative Strength: Calculated Goal Difference (GF - GA) as a proxy for team quality.

Efficiency Signals: Shot Accuracy (Shots on Target / Total Shots) to capture clinical finishing.

📈 Modeling Strategy
I utilized Multinomial Logistic Regression with an lbfgs solver.

Time-Series Validation: Used a rolling window approach where all previous seasons were used for training to predict the next single season.

Standardization: Applied StandardScaler fitted only on training data to prevent leakage.

🧪 Model Evaluation
Since the objective is probabilistic accuracy rather than just "guessing the winner," the following metrics were used:

Log Loss: Measures the uncertainty of the predictions. The model achieved a consistent Log Loss between 0.95 and 1.01 across different seasons.

Brier Score: Evaluated the accuracy of probabilities for each specific class (H, D, A).

Probability Calibration: Analyzed through calibration curves to determine if a predicted 70% probability actually results in a win 70% of the time.

💡 Key Insights & Findings
Model Calibration: The model is well-calibrated for low-to-mid range probabilities across all outcomes.

Over-optimism in Away Wins: Evaluation showed that the model tends to overestimate Away wins (Class 'A') at high probability thresholds, suggesting a need for specialized recalibration for away games.

How to Run
Clone the repository.

Install dependencies: pip install pandas numpy scikit-learn matplotlib.

Open and run the Jupyter Notebook: Multiclass_probabilistic_classification_football.ipynb.

This project was developed as a demonstration of end-to-end Machine Learning workflows, specifically focusing on time-series data and probabilistic evaluation.
