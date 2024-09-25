### PremierLeagueAI - Model Documentation (Version 0.1)

#### Project Overview:
The PremierLeagueAI project is focused on building machine learning models that predict whether a football team will win a particular match based on various historical and match-specific features. The ultimate goal is to provide insights into match outcomes for football teams in the English Premier League (EPL), using machine learning to analyze match performance and rolling averages of match statistics.

---

### 1. **Problem Definition:**
The primary objective is to predict whether a football team will win a given match. This is treated as a binary classification problem where:
- **1 (Win)**: Indicates that the team won the match.
- **0 (Not Win)**: Indicates that the team either lost or drew the match.

The model outputs a probability score for a win and classifies the match outcome based on the highest probability.

---

### 2. **Data Overview:**
The data used in this model contains historical match records from the Premier League. The dataset consists of features that describe the context of each match, team performance, and rolling averages for key match statistics.

**Dataset Details:**
- **Data Source**: Preprocessed data containing historical match results.
- **Data Range**: Data spans from 2021 to 2024.
- **Rows**: 2278 rows (matches).
- **Columns**: 40 columns (including both raw match details and engineered features).

---

### 3. **Features Used:**
The dataset includes various features describing match-level information and team performance. Here are the key features used in building the machine learning models:

#### 3.1. **Basic Match Information:**
1. **venue_code**: Encoded feature representing whether the team played at home or away. (Categorical: 0 = Away, 1 = Home)
2. **opp_code**: Encoded representation of the opponent team.
3. **hour**: The hour at which the match was played (derived from the `time` column).
4. **day_code**: The day of the week on which the match was played (0 = Monday, 6 = Sunday, derived from `date` column).

#### 3.2. **Rolling Averages:**
These features are computed as rolling averages for each team’s last three matches. The purpose of using rolling averages is to capture a team's recent performance trends:
1. **gf_rolling**: Rolling average of goals scored (goals for).
2. **ga_rolling**: Rolling average of goals conceded (goals against).
3. **sh_rolling**: Rolling average of shots taken by the team.
4. **sot_rolling**: Rolling average of shots on target.
5. **dist_rolling**: Rolling average of shooting distance.
6. **fk_rolling**: Rolling average of free kicks.
7. **pk_rolling**: Rolling average of penalty kicks.
8. **pkatt_rolling**: Rolling average of penalty kick attempts.

#### 3.3. **Target Variable:**
- **target**: This is the binary target variable indicating whether the team won the match or not:
  - **1**: Team won.
  - **0**: Team did not win (either a draw or loss).

---

### 4. **Modeling Approach:**
Two models were developed to predict the match outcome (Win/Not Win):

#### 4.1. **RandomForestClassifier:**
- **Model Overview**: A Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.
- **Parameters**:
  - `n_estimators=100`: The model consists of 100 decision trees.
  - `random_state=42`: For reproducibility.
  
- **Why Chosen**: Random Forest was chosen for its ability to handle complex data structures and its robustness in reducing overfitting by averaging the results of multiple trees.

#### 4.2. **LogisticRegression:**
- **Model Overview**: Logistic Regression is a linear model that estimates the probability of a binary outcome (win or not win) using a logistic function.
- **Parameters**:
  - `max_iter=1000`: The number of iterations for the solver to converge.
  - `random_state=42`: For reproducibility.

- **Why Chosen**: Logistic Regression was chosen due to its simplicity and interpretability. It helps provide a baseline for comparison with more complex models like Random Forest.

---

### 5. **Model Training and Evaluation:**

#### 5.1. **Train-Test Split**:
- The dataset is split into **80% training** data and **20% test** data to evaluate model performance.

#### 5.2. **Metrics Used**:
The following metrics were used to evaluate model performance on the test set:
- **Precision**: Measures the accuracy of positive predictions (Wins).
- **Accuracy**: Overall accuracy of the model across all predictions.
- **ROC AUC (Receiver Operating Characteristic - Area Under the Curve)**: Measures the model's ability to distinguish between positive and negative classes. A higher value indicates better performance.

---

### 6. **Model Results:**
Here are the evaluation results for both models:

#### 6.1. **RandomForestClassifier:**
- **Precision**: 0.5631
- **Accuracy**: 0.6162
- **ROC AUC**: 0.6046

The RandomForest model has decent predictive power but room for improvement, with an ROC AUC score slightly above random guessing (0.5). Its precision shows that about 56.3% of predicted wins were correctly identified as wins.

#### 6.2. **LogisticRegression:**
- **Precision**: 0.5934
- **Accuracy**: 0.625
- **ROC AUC**: 0.6345

The LogisticRegression model performed marginally better than RandomForest in terms of ROC AUC (0.6345) and precision (59.34%). However, both models show similar overall accuracy levels.

---

### 7. **Model Logging with MLflow:**
All models and their respective metrics were logged using **MLflow**, enabling experiment tracking and version control of models.

- **Parameters Logged**: Model hyperparameters, including `n_estimators` for RandomForest and `max_iter` for LogisticRegression.
- **Metrics Logged**: Precision, accuracy, and ROC AUC for both models.
- **Model Artifacts Saved**: The trained models were saved as `.joblib` files for future inference or deployment.

---

### 8. **Recommendations for Improvement:**
1. **Feature Engineering**: Adding more features related to player statistics, team dynamics, and match context (e.g., weather conditions, injuries) could enhance the model’s predictive power.
2. **Hyperparameter Tuning**: Further tuning of hyperparameters using techniques like `GridSearchCV` or `RandomizedSearchCV` could lead to improvements in model performance.
3. **More Complex Models**: Explore advanced algorithms like **Gradient Boosting Machines (GBM)**, **XGBoost**, or **LightGBM** to see if these models provide better predictive power.
4. **Handling Class Imbalance**: Investigate whether there’s a class imbalance between wins and losses, and if so, apply techniques like oversampling, undersampling, or class weighting to improve model performance.
5. **Model Calibration**: Calibrating the probability outputs from models to improve probability estimates and potentially increase ROC AUC scores.

---

### 9. **Conclusion:**
Both models have demonstrated moderate predictive power, with LogisticRegression slightly outperforming RandomForest. However, there is room for improvement in terms of feature engineering, hyperparameter optimization, and model complexity. Future work will focus on enhancing the model’s predictive accuracy and generalizability, potentially exploring more advanced techniques and expanding the feature set.