# Premier League Match Outcome: Multiclass Probabilistic Classification

This project implements a **multiclass probabilistic classification** model to predict the outcomes of English Premier League matches (Home Win, Draw, Away Win). Instead of simple labels, the model focuses on generating **calibrated probabilities**, which are crucial for assessing risk and value in sports analytics.

## 📌 Project Overview
The goal is to predict the probability of the three possible outcomes (H, D, A) using historical match data while strictly avoiding **data leakage**. The project employs a **rolling season-by-season training strategy** to simulate real-world conditions where the model is updated as new data becomes available.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Logistic Regression, StandardScaler)
* **Metrics & Evaluation:** `Log Loss`, `Brier Score`, `Calibration Curves`
* **Visualization:** `matplotlib`

## 📊 Data & Feature Engineering
The dataset consists of Premier League matches from seasons **2018/19 to 2023/24**, sourced from [football-data.co.uk](https://www.football-data.co.uk/).

## 🚀 Key Features
- **Advanced Feature Engineering**: Utilizes Exponential Moving Averages (EMA) with a span of 5 matches to capture recent team form, ensuring zero data leakage through proper time-shifting.
- **Rolling Season Training**: Simulates real-world deployment by training on all previous seasons to predict the current one.
- **Probabilistic Evaluation**: Uses **Log Loss** and **Brier Scores** to measure the quality of the predicted probabilities.
- **Calibration Analysis**: Includes Reliability Diagrams (Calibration Curves) to detect and visualize model bias.
- **Value Betting Simulator**: A backtesting engine that calculates ROI and bankroll evolution based on a "Value" strategy against Bet365 odds.
- **Dark-Theme Visualizations**: High-quality plots for professional presentation.

## 📊 Methodology

### 1. Data Pipeline
Data is fetched from `football-data.co.uk`, covering multiple Premier League seasons (2018-2024). 
- **Features**: Home/Away Attack Form, Home/Away Defense Form (calculated via EMA).
- **Target**: Match Result (H: 0, D: 1, A: 2).

<img width="990" height="590" alt="image" src="https://github.com/user-attachments/assets/f845efee-1f85-4f21-9616-0e9aa099ca17" />


### 2. The Model
A **Multinomial Logistic Regression** model is employed to provide well-behaved probability estimates across the three possible outcomes.

### 3. Calibration
Recognizing that models can be over-confident (especially for Away wins), the project includes an optional **Isotonic Regression** calibration step using `CalibratedClassifierCV`.

<img width="1427" height="948" alt="image" src="https://github.com/user-attachments/assets/1531f81b-f5e7-4b04-b1b5-d7fd16f24938" />


## 📈 Performance & Backtesting
The model was evaluated on the 2023/2024 season using a 1% value threshold.

<img width="986" height="471" alt="image" src="https://github.com/user-attachments/assets/019298c8-feb8-49c4-b75b-273cadba6ec7" />


- **Total Bets Placed**: 1,787
- **ROI**: -4.76%
- **Final Bankroll**: 95.24 (from initial 100)

*Note: The ROI reflects the difficulty of beating the bookmaker's margin (overround). The stability of the bankroll curve suggests the model effectively captures market dynamics but requires further edge refinement.*


## 🔮 Future Improvements
- Integration of Expected Goals (xG) data for better performance tracking.

- Implementing Kelly Criterion for dynamic bet sizing.

- Testing non-linear models like XGBoost or LightGBM.

- Inclusion of Market Closing Odds to measure "Beat the Close" performance.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib jupyter
    ```
3.  **Run the analysis:**
    Open and run the Jupyter Notebook: `Multiclass_probabilistic_classification_football.ipynb`


---
*This project was developed as a demonstration of end-to-end Machine Learning workflows, specifically focusing on time-series data and probabilistic evaluation.*
