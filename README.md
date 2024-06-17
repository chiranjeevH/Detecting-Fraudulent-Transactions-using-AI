# AI-Driven Financial Fraud Detection

## Project Overview
**Problem Statement:** The rise of digital transactions has led to an increase in financial frauds, which are often sophisticated and difficult to detect with traditional rule-based systems. This project aims to leverage advanced machine learning techniques to enhance the detection capabilities of fraudulent activities in financial transactions.

**Objective:** Develop a predictive model that effectively identifies and flags fraudulent transactions using machine learning techniques, thereby aiding financial institutions in preventing potential financial losses.

## Data Description
- **Dataset:** The project utilizes the Kaggle Credit Card Fraud Detection dataset, which consists of transactions made by credit cards over two days in September 2013 by European cardholders.
- **Features:** The dataset contains 284,807 transactions, out of which 492 are frauds. Features include numerical input variables which are the result of a PCA transformation due to confidentiality issues. The features are labeled V1, V2, ... V28.

## Methodology
### Data Preprocessing
- **Feature Scaling:** Applied scaling on the 'Amount' feature to normalize the range of transaction amounts.
- **Handling Imbalance:** The dataset was highly imbalanced. Techniques such as under-sampling and over-sampling were evaluated to equilibrate the classes.

### Exploratory Data Analysis (EDA)
- **Visualization:** Conducted various visualizations to understand the nature of fraudulent and non-fraudulent transactions using histograms, box plots, and scatter plots.

### Model Development
- **Machine Learning Models Used:**
  - **XGBoost Classifier:** For its efficiency and effectiveness at handling imbalanced data.
  - **K-Means Clustering:** Applied to detect anomalies as a preprocessing step.
  - **Neural Networks:** To capture non-linearities in the data.
- **Feature Importance:** Analyzed using XGBoost to identify the most critical features contributing to fraud.

### Model Evaluation
- **Metrics:** Focused on Precision, Recall, and AUC-ROC to evaluate model performance due to the imbalanced nature of the dataset.
- **Validation:** Applied cross-validation techniques to ensure the model's generalizability.

## Results
- **Performance:** Achieved a recall of 76%, precision of 94%, and AUC-ROC of approximately 97% on the test set, indicating high effectiveness in identifying fraudulent transactions.
- **Insights:** The model identified V14, V4, and V10 as the top predictors of fraud.

## Technologies Used
- **Python** for all data processing and model development, utilizing libraries such as Pandas, NumPy, Scikit-learn, XGBoost, and Keras.
- **Matplotlib** and **Seaborn** for data visualization.

## Conclusion
This project demonstrates the potential of machine learning in enhancing fraud detection systems. Future work will focus on integrating the model into a real-time transaction processing system and exploring the use of more complex ensemble methods to further improve the detection rates.

## Dataset
The dataset used for this project is available on Kaggle: [Paysim Synthetic Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1). It provides a comprehensive collection of synthetic financial transactions, making it an ideal resource for training and evaluating our fraud detection models.

## Dependencies
This project requires the following dependencies:
- `scikit-learn` for implementing machine learning algorithms
- `numpy` and `pandas` for data manipulation and preprocessing
- `matplotlib` for generating visualizations
- `xgboost` for advanced boosting algorithms
- `pickle` for saving and loading models
- `psutil` for monitoring system memory
- Custom functions from the `GAN.py` file
- Additional Python libraries for various AI algorithms and methods

## How to Run
1. Ensure you have all the necessary dependencies installed using `pip` or `conda`.
2. Clone or download the project repository to your local machine.
3. Open the Jupyter notebook or Python script containing your project code.
4. Run the code cells or script to execute the AI algorithms, train the models, and evaluate their performance.

## Contributions and Feedback
We welcome contributions, feedback, and collaborations from the community. Feel free to open issues, submit pull requests, or contact any of the team members listed above.
