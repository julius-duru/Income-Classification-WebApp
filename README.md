https://income-classification-webapp.streamlit.app/

Income Classification Project
Project Overview
This project focuses on building a binary classification model to predict whether an individual earns more than $50K per year or less based on demographic, educational, and employment-related features. The project follows a complete data science workflow, from exploratory data analysis (EDA) and data preprocessing to feature engineering, visualization, and model-ready insights.
The dataset reflects real-world socioeconomic data, where challenges such as missing values, skewed distributions, and class imbalance are addressed using principled and explainable techniques.
Problem Statement
Accurately predicting income level is valuable for:
•	Socioeconomic analysis
•	Policy planning
•	Financial and marketing segmentation
•	Understanding drivers of economic inequality
The goal is to learn patterns that distinguish individuals earning less than 50K from those earning more than 50K, while ensuring the model is interpretable and robust.
Dataset Description
•	Total records: 32,561
•	Target variable: income
o	<=50K
o	>50K
•	Feature types:
o	Numerical: age, education-num, capital-gain, capital-loss, hours-per-week
o	Categorical: workclass, occupation, marital-status, relationship, race, sex, native-country
Data Preprocessing and Cleaning
Handling Missing Values
Certain categorical features (workclass, occupation, native-country) contained missing values originally encoded as "?".
These were:
•	Replaced with NaN
•	Subsequently encoded as a distinct category "Unknown"
This approach preserves potentially informative missingness rather than discarding data especially as the total occurrence in each column was more than 5%.
Exploratory Data Analysis (EDA)
1. Income Class Balance
A bar chart of the target variable shows a moderate class imbalance, with the <=50K class being more frequent. In future, evaluation metrics beyond accuracy (e.g., F1-score, ROC-AUC) will be used on this dataset.
2. Age Distribution by Income
Histogram and KDE plots reveal:
•	Higher-income individuals tend to be older, peaking in the late 30s to mid-40s
•	Lower-income individuals are more concentrated in younger age ranges
•	Significant overlap exists, indicating age alone is insufficient for classification
Age is a strong but non-decisive predictor, reinforcing the need for multivariate modeling.
3. Hours-per-week vs Income
A violin plot comparing weekly working hours shows:
•	>50K earners generally work longer hours
•	Greater variability and overtime patterns among higher-income individuals
•	Evidence of underemployment in the lower-income group
This highlights work intensity as a meaningful socioeconomic signal.
4. Capital Gain Indicator vs Income
Capital gains are rare but highly informative:
•	Individuals with any capital gain are disproportionately represented in the >50K class
•	A binary feature (capital_gain) was engineered to capture this effect
5. Correlation Analysis
A correlation heatmap of numeric and encoded features shows:
•	Strong associations between education-num, capital-gain, hours-per-week, and income
•	Weak linear correlations for some categorical encodings, justifying the use of non-linear models
Feature Engineering
Key feature transformations include:
•	Binary indicators for capital gain and capital loss
•	Log transformations for highly skewed financial variables
•	Encoding missing categorical values as "Unknown"
These steps improved both model performance and interpretability.
Modeling Approach
While the focus of this project is on EDA and feature understanding, the data preparation is optimized for:
•	Logistic Regression (baseline, interpretable)
•	Tree-based models (Random Forest, XGBoost, CatBoost)
o	Particularly effective due to non-linear relationships and mixed feature types
Tree-based models are especially suitable given:
•	Skewed distributions
•	Interaction effects
•	Informative missingness
Key Insights
•	Income is strongly influenced by age, education level, work intensity, and capital gains
•	Missing data carries predictive signal, hence were not removed
•	Socioeconomic patterns are non-linear and require robust feature interactions
Modeling and Results
Multiple machine learning algorithms were trained and evaluated to identify the most effective approach for income classification. Both training accuracy and cross-validation performance were considered to ensure generalization and avoid overfitting.
Model Performance Summary
Model	Evaluation Metric	Score
Logistic Regression	Training Accuracy	0.8222
K-Nearest Neighbors (KNN)	Accuracy	0.8284
Support Vector Classifier (SVC)	Accuracy	0.8284
Random Forest	Best CV Accuracy	0.8283
	OOB Score	0.8242
Best Overall Model	Cross-Validation Accuracy	0.8348

Interpretation of Results
•	Logistic Regression provided a strong and interpretable baseline, achieving over 82% accuracy, which confirms that the dataset contains meaningful linear relationships.
•	KNN and SVC slightly outperformed Logistic Regression, indicating the presence of non-linear decision boundaries within the data.
•	Random Forest demonstrated robust performance with:
o	Competitive cross-validation accuracy
o	A strong Out-of-Bag (OOB) score, confirming good generalization without excessive overfitting
•	The best cross-validation accuracy (~83.5%) suggests that ensemble or tuned non-linear models are most suitable for this problem.
Overall, results confirm that income prediction is a non-linear classification problem, where ensemble and margin-based models outperform linear approaches.
Further Insight
Although multiple models achieved similar accuracy, the dataset will be tested using tree-based and kernel-based models. This is in order to: 
•	Capture feature interactions
•	Robustness to skewed distributions
•	Have reduced sensitivity to feature scaling and outliers
Conclusion
This project demonstrates an end-to-end classification workflow with strong emphasis on:
•	Thoughtful data cleaning
•	Insight-driven visualizations
•	Feature engineering grounded in domain understanding
The modeling phase demonstrates:
•	Strong baseline performance
•	Consistent improvement with non-linear models
•	Reliable generalization validated through cross-validation and OOB scoring
These results validate the quality of the preprocessing, feature engineering, and exploratory analysis conducted earlier in the project. The resulting dataset and insights provide a solid foundation for building accurate and interpretable income prediction models.

