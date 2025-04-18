### env (WSL)
#  source ~/anaconda3/bin/activate
#  conda activate ml_class
###

# Building stress detection algorithm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np


#import seaborn as sns

# train_data = pd.read_csv('final/train.csv')
# test_data = pd.read_csv('final/test.csv')

# Strip column names (in case of spaces)
# train_data.columns = train_data.columns.str.strip()

# # Convert to numeric (handles non-numeric values)
# # The line `train_data["MEAN_RR"] = pd.to_numeric(data["MEAN_RR"], errors="coerce")` is converting the
# # values in the "MEAN_RR" column of the `train_data` DataFrame to numeric data type.
# train_data["MEAN_RR"] = pd.to_numeric(train_data["MEAN_RR"], errors="coerce")
# train_data["HR"] = pd.to_numeric(train_data["HR"], errors="coerce")

# # Drop missing values
# data = train_data.dropna(subset=["MEAN_RR", "HR"])

def preprocess_data_supervised(file_path):
    df = pd.read_csv(file_path)

    # condition_encoding = {'no stress': 0, 'time pressure': 1, 'interruption': 2}
    new_binary_encoding = {'no stress': 0, 'stress': 1}
    #df['condition'] = df['condition'].map(condition_encoding)
    df['condition'] = df['condition'].replace({'interruption': 'stress', 'time pressure': 'stress'})
    df['condition'] = df['condition'].map(new_binary_encoding)

    selected = df[["MEAN_RR", "MEDIAN_RR", "SDRR", "SDRR_RMSSD", "HR", "MEAN_REL_RR", "MEDIAN_REL_RR", "SDRR_REL_RR"]]
    features = selected.iloc[:]
    target = df.iloc[:, -1]

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    pca = PCA(n_components=0.95, random_state=42)
    pca_features = pca.fit_transform(standardized_features)

    return pca_features, target

def evaluate_classification(model, train_x, train_y, test_x, test_y):
    y_pred = model.predict(test_x)
    report = classification_report(test_y, y_pred)
    print(report)
    
    # cv_scores = cross_val_score(model, train_x, train_y, cv=10, scoring='accuracy')
    # print('cv_scores.mean:', f'{cv_scores.mean():.4f}')
    
    # Compute confusion matrix
    cm = confusion_matrix(testy, model.predict(testx))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add labels
    classes = ["No Stress", "Stress"]  # Adjust as needed
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

trainx, trainy = preprocess_data_supervised('train.csv')
testx, testy = preprocess_data_supervised('test.csv')

random_forest = RandomForestClassifier(
    class_weight= 'balanced',
    criterion='gini',
    max_depth=8,
    max_features='log2',
    n_estimators=120
)

decision_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    max_features='log2'
)

# decision_tree.fit(trainx, trainy)

# evaluate_classification(decision_tree, 
#                         trainx, trainy, testx, 
#                         testy)

random_forest.fit(trainx, trainy)

evaluate_classification(random_forest, 
                        trainx, trainy, testx, 
                        testy)



#graphing a pie chart
# df = pd.read_csv('final/test.csv')

# # merge 2 classes into stress class
# df['condition'] = df['condition'].replace({'interruption': 'stress', 'time pressure': 'stress'})


# condition_counts = df['condition'].value_counts()
# labels = condition_counts.index
# sizes = condition_counts.values
# plt.pie(sizes, labels=labels, autopct='%1.1f%%')
# plt.title('condition_counts')
# plt.show()