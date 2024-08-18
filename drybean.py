# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import ConfusionMatrixDisplay

from ucimlrepo import fetch_ucirepo

# Fetch dataset
df = fetch_ucirepo(id=602)

# Data (as pandas dataframes)
X = df.data.features
y = df.data.targets.values.ravel()

# Convert X and Y to numpy array
X_npArray = np.array(X)
y_npArray = np.array(y)

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Feature Importances with ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=500, random_state=42)
model.fit(X, y)
print(model.feature_importances_)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

# Plot feature importances
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

# Standardizing Features
scaler_X = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)


# Define function to plot learning curves
def plot_learning_curves(estimator, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Learning Curves ({estimator.__class__.__name__})")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.legend(loc='best')
    plt.show()

# Model Training and Learning Curves for KNN
knnModel = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', p=2)
knnModel.fit(X_train_scaled, y_train)
y_pred = knnModel.predict(X_test_scaled)

# Classification Report and Accuracy Score for KNN
print('KNN Accuracy: %.5f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=np.unique(labelencoder.inverse_transform(y))))

# Classification Visualization Using ConfusionMatrixDisplay
titles_options = [
    ("Confusion matrix, without normalization", None, '.0f'),
    ("Normalized confusion matrix", "true", '.2f'),
]
for title, normalize, values_format in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        knnModel,
        X_test_scaled,
        y_test,
        display_labels=np.unique(labelencoder.inverse_transform(y)),
        cmap=plt.cm.Oranges,
        normalize=normalize,
        xticks_rotation='vertical',
        values_format=values_format
    )
    disp.ax_.set_title(title)

plt.show()

# Plot learning curves for KNN model
plot_learning_curves(knnModel, X_train_scaled, y_train)

# EXTRA TASK: Using Support Vector Machines (SVM) model
svmClassModel = svm.SVC()
svmClassModel.fit(X_train_scaled, y_train)
y_pred_forSVM = svmClassModel.predict(X_test_scaled)

# Classification Report and Accuracy Score for SVM
print('SVM Accuracy: %.5f' % accuracy_score(y_test, y_pred_forSVM))
print(classification_report(y_test, y_pred_forSVM, target_names=np.unique(labelencoder.inverse_transform(y))))

# Plot learning curves for SVM model
plot_learning_curves(svmClassModel, X_train_scaled, y_train)