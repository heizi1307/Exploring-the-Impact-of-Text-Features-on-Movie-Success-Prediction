Used Packages:
scikit-learn: Used for machine learning algorithms such as Naive Bayes, Decision Trees, K-Nearest Neighbors, and Random Forest.
numpy: Used for numerical operations and array manipulations.
pandas: Used for data manipulation and analysis, especially for reading and processing CSV files.
matplotlib: Used for data visualization, including plotting accuracy charts.

Variable Naming:
non_text_features: List of non-textual features used in the analysis.
train_file, eva_file, test_file: DataFrames containing training, evaluation, and test data, respectively.
X_train_without_text, y_train_rating, y_train_cate: Training features and target variables.
X_eval_without_text, y_eval_rating, y_eval_cate: Evaluation features and target variables.
X_test_without_text: Test features.
mode_values: Mode values used for filling missing values in non-text features.

Function Input and Output:
Function: test_acc(x_train, y_train, x_test, y_test)
Input:
x_train, x_test: Features for training and testing.
y_train, y_test: Target variables for training and testing.
Output:
Accuracy scores of different classifiers on the test data.
Visualization showing the accuracy of different classifiers.

Run code by this order:
Reading Files:
The training, evaluation, and test datasets are read from CSV files.
Original language data is encoded using LabelEncoder to prepare it for model training.
Data Preprocessing:
Non-text features like release year, runtime, budget, etc., are extracted for further processing.
Missing values in non-text features are handled by filling them with mode values.
Statistics about production companies are analyzed, including counts, empty values, and top 5 companies.
Model Training and Evaluation:
Several classifiers are trained and evaluated using different sets of features.
The following classifiers are used:
Multinomial Naive Bayes
Decision Tree
K-Nearest Neighbors
Random Forest
Self-Training Classifier
Results and Analysis:
Model accuracies are measured and compared using accuracy scores and visualizations.
Classification reports are generated to provide detailed insights into model performance.
Baseline models (e.g., random baseline) are also evaluated to establish performance benchmarks.
Feature Selection:
Feature selection techniques such as SelectKBest with chi-squared scoring are applied to identify the most relevant features.
The impact of different numbers of selected features on model performance is analyzed and visualized.
Self-Training:
Self-Training classifiers are employed to leverage both labeled and unlabeled data for improved performance.
Both Decision Tree and Multinomial Naive Bayes classifiers are used with Self-Training technique.
Kaggle Submission:
The best-performing model is used to predict rating categories for the test dataset, and results are saved to a CSV file for Kaggle submission.
Visualizations:
Accuracies of different models across various tests are visualized using bar charts.
Accuracy trends with varying numbers of selected features are plotted to understand feature impact on model performance.

File name A2.ipynb is the code file. When running the code, please put the csv file and the code file in the same directory, 
and the BoW and TFIDF files unzip and put the folder in the same directory as the code. For example, the absolute path to 
access "train_production_companies_bow.npz" from the code is: 'TMDB_text_features_bow/train_production_companies_bow. npz'

Experiment Results:
Decision Tree (DT) model outperforms Multinomial Naive Bayes (MNB) and K-Nearest Neighbors (KNN) models across various feature data sets, including production companies (PC) data and concatenated data.
All models exhibit significant improvement over the random baseline, indicating their ability to learn and make better predictions.
DT's superiority can be attributed to its capability to handle complex relationships and outliers effectively, compared to MNB and KNN.
Feature analysis reveals the importance of proper handling of empty values, especially in concatenated data, for improved model performance.
Semi-supervised learning with DT shows slight performance improvement, while MNB's performance remains largely unaffected.
Random Forests (RF) effectively address class imbalances, leading to notable improvements in prediction accuracy.
Feature selection analysis highlights the subtle effects of different feature sets on model performance, with DT showing slight improvement with increasing features.