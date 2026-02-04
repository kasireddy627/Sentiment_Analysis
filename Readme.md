# Sentiment Analysis of Flipkart Product Reviews

An end-to-end Machine Learning project that analyzes Flipkart product reviews and predicts customer sentiment (Positive / Negative).
The project covers the complete ML lifecycle — from data loading and analysis to model training, evaluation, selection, and deployment on AWS EC2 using Flask.


## Project Overview

Customer reviews are a critical source of feedback for e-commerce platforms.
This project builds an automated sentiment analysis system that classifies Flipkart product reviews into **Positive** or **Negative** sentiments, enabling businesses to:

* Identify customer satisfaction and dissatisfaction
* Detect negative feedback early
* Improve product quality and user experience

The final solution is deployed as a **web application** accessible via a public EC2 URL.

## Project Workflow

    STEP 1: Load data
    STEP 2: Create sentiment labels
    STEP 3: Clean text (remove noise + stopwords)
    STEP 4: Analyze cleaned text (insights)
    STEP 5: Train-test split
    STEP 6: Multiple models
    STEP 7: Pick best (F1-score)
    STEP 8: Deploy into AWS

## Web Application Screenshots

![Web Application Screenshot](templates/Static/image1.png)

![Web Application Screenshot](templates/Static/image2.png)

 
## Dataset Description

* **Source**: Flipkart product reviews dataset
* **Total Records**: 8,518
* **Columns**:

  * Reviewer Name
  * Review Title
  * Place of Review
  * Up Votes
  * Down Votes
  * Month
  * Review Text
  * Ratings

### Sentiment Labeling Logic

| Rating | Sentiment                             |
| ------ | ------------------------------------- |
| ≥ 4    | Positive                              |
| ≤ 2    | Negative                              |
| 3      | Excluded (to maintain binary clarity) |



## Exploratory Data Analysis (EDA)

### Class Distribution

* Positive: ~80%
* Negative: ~20%

This imbalance required careful model evaluation beyond accuracy, focusing on:

* Recall for the Negative class
* Macro and Weighted F1-scores

### Key Insights

* Positive reviews frequently mention quality, value, and satisfaction
* Negative reviews emphasize defects, poor quality, and dissatisfaction


## Text Preprocessing

Each review is cleaned using the following steps:

1. Remove special characters and digits
2. Convert text to lowercase
3. Tokenize text
4. Remove stopwords
5. Lemmatize words

This ensures consistent, noise-free input for feature extraction.



## Feature Engineering

* **Technique**: TF-IDF Vectorization
* **Configuration**:

  * Unigrams and bigrams
  * Maximum features: 5,000

TF-IDF helps capture both important words and meaningful phrases while reducing the impact of common terms.



## Model Training & Experimentation

Multiple models were trained and evaluated:

* Logistic Regression (Baseline)
* Logistic Regression (Class-weighted / Balanced)
* Logistic Regression + SMOTE
* Linear SVM (SVC)
* Naive Bayes
* Decision Tree + SMOTE

### Evaluation Metrics

* Accuracy
* Weighted F1-score
* Macro F1-score
* Recall for Negative class



## Model Comparison Summary

| Model                          | Accuracy | Weighted F1 | Macro F1 | Negative Recall |
| ------------------------------ | -------- | ----------- | -------- | --------------- |
| SVM (Linear)                   | 0.869    | 0.859       | 0.765    | 0.513           |
| Logistic Regression (Balanced) | 0.858    | 0.859       | 0.780    | **0.662**       |
| Logistic Regression (Baseline) | 0.871    | 0.854       | 0.748    | 0.436           |
| Decision Tree + SMOTE          | 0.854    | 0.843       | 0.738    | 0.478           |
| Naive Bayes                    | 0.860    | 0.833       | 0.703    | 0.338           |
| Logistic Regression + SMOTE    | 0.700    | 0.729       | 0.643    | 0.763           |


## Final Model Selection

**Balanced Logistic Regression** was chosen as the final model because:

* Strong overall F1-score
* Significantly better recall for Negative reviews
* Interpretable and production-friendly
* Avoids overfitting introduced by aggressive resampling



## Model Pipeline

The final model is implemented as a Scikit-learn Pipeline:

* TF-IDF Vectorizer
* Logistic Regression with `class_weight="balanced"`

The entire pipeline is serialized using `pickle` for deployment.


## Architecture Diagram

```text
+-------------------+
|   User (Browser)  |
+---------+---------+
          |
          v
+-------------------+
|   Flask Web App   |
|  (HTML + API)     |
+---------+---------+
          |
          v
+-------------------+
|  Pickled ML Model |
| (TF-IDF + LR)     |
+---------+---------+
          |
          v
+-------------------+
|  Prediction Output|
| Positive / Negative |
+-------------------+
```


## Web Application (Flask)

### Features

* Input box for entering product reviews
* Real-time sentiment prediction
* Project overview and model explanation
* Model comparison metrics displayed on UI

### API Endpoints

* `/` → Home page
* `/predict` → Returns sentiment prediction


## Conclusion

This project demonstrates a complete production-grade ML workflow:

* Data understanding and preprocessing
* Handling class imbalance effectively
* Systematic model experimentation
* Justified model selection
* Real-world deployment on AWS