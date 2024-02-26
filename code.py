import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV  
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter

# Data analysis and classification distribution
def plot_images(images, labels, nrows=5, ncols=5):          
    num_images = len(images)
    num_displayed = min(num_images, nrows * ncols)                 # Pick number of images to show
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4))    # Create a figure and a set of subplots
    for i in range(num_displayed):
        ax = axes.flat[i]
        ax.imshow(images[i].reshape(20, 20), cmap='gray')       # Display data as an image
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')

# load dataset
X = np.load("emnist_hex_images.npy")
y = np.load("emnist_hex_labels.npy")

# Split data before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the dataset using MinMaxScaler, fit on training data and transform both training and test data
scaler = MinMaxScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class distribution on train data
labels_df = pd.DataFrame(y_train, columns=["Labels"])         # Convert to pandas dataframe
class_distribution = labels_df["Labels"].value_counts()       # Count number of occurrences of labels
print("Class Distribution on Training Data:", class_distribution)

# Plot class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.title("Class Distribution on Training Data")
plt.xticks(rotation='horizontal')
plt.show()

# Grid search to find best set of parameters

random_seed = np.random.randint(1, 9)                        # Pick a random integer to select slice

# Define parameter grids for each classifier
param_grids = {
    'MLPClassifier': {
        'hidden_layer_sizes': [(400,), (400, 200),],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01],
        'learning_rate': ['adaptive'],
        'batch_size': [64],
        'max_iter': [1000],
        'early_stopping': [True],
        'learning_rate_init': [0.001],
    },
    'SVC': {
        'C': [10],  # Regularization parameter
        'kernel': ['poly', 'rbf'],  # Kernel type
        'degree': [2],
        'gamma': [10],  # Kernel coefficient
        'coef0': [0.1],
        'class_weight': [None],
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5,],
        'weights': ['distance'],
        'algorithm': ['auto',],
        'leaf_size': [25],
        'n_jobs': [-1],
    }
}

# Store classifiers
classifiers = {
    'MLPClassifier': MLPClassifier(random_state=42),
    'SVC': SVC(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier()
}

best_f1_score = 0
best_model = None
best_params = None
cv_results_dict = {}

inter_s = (random_seed-1)*10000 
inter_f = random_seed*10000
# Iterate over classifiers and corresponding parameter grids
for classifier_name, classifier in classifiers.items():
    param_grid = param_grids[classifier_name]
    
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='f1_weighted', cv=5)      
    grid_search.fit(X_train[inter_s:inter_f], y_train[inter_s:inter_f])        # Fit to random slice of training data
    
    cv_results_dict[classifier_name] = grid_search.cv_results_               # Store results of each classifier



cls_scores = {}
# Iterate over classifiers and their results
for classifier_name, cv_result in cv_results_dict.items():
    
    cls_scores[classifier_name] = []
    for i in range(5):
        fold_score = cv_results_dict[classifier_name][f'split{i}_test_score'].max()       # Pick the best score on each fold of every classifier 
        cls_scores[classifier_name].append(fold_score)

best_scores = {}   
# Find the best score on each fold over all classifiers
for i in range(5):
    best_score = 0         # Initialize best score for the current fold
    best_model = None       # Initialize best model for the current fold
    
    # Iterate over the scores of the classifiers
    for classifier_name in cls_scores.keys():
        clf_score = cls_scores[classifier_name][i]
        
        if clf_score > best_score:
            best_score = clf_score
            best_model = classifier_name
    best_scores[i] = (best_score, best_model)          # Store the score and the type of classifier
    


# Extract the classifier names and scores from the best_scores dictionary
classifier_scores = [(score, best_model) for score, best_model in best_scores.values()]
classifier_names = [best_model for _, best_model in classifier_scores]

# Count the occurrences of each classifier
classifier_counts = Counter(classifier_names)

# Find the classifier(s) with the maximum count (majority classifier)
majority_classifiers = [classifier for classifier, count in classifier_counts.items() if count == max(classifier_counts.values())]

# Find the best score of the majority classifier
majority_classifier_scores = [score for score, classifier in classifier_scores if classifier in majority_classifiers]
best_score_majority_classifier = max(majority_classifier_scores)

# If there is a single majority classifier, it will be in majority_classifiers[0]
# If there are multiple classifiers tied for the majority, they will all be in majority_classifiers
if len(majority_classifiers) == 1:
    majority_classifier = majority_classifiers[0]
    print(f"The majority classifier is: {majority_classifier}")
    print(f"The best score of the majority classifier is: {best_score_majority_classifier}")
else:
    print(f"There is a tie among the majority classifiers: {majority_classifiers}")
    print(f"The best score of the majority classifiers is: {best_score_majority_classifier}")

# Test classifier on the test set
clf = classifiers[majority_classifier]
clf.fit(X_train[:], y_train[:])                 # Fit the best classifier to the whole training set
predictions = clf.predict(X_test[:])        # Make predictions on unseen data
score = f1_score(predictions, y_test[:], average='weighted')      # Compute f1-score

print(score)

empty_label = 16
empty_images = []

# Store predicted empty images and visualise some of them
for idx, pred_label in enumerate(predictions):
    if pred_label == empty_label:
        empty_images.append(X_test[idx])
        
plot_images(empty_images, [empty_label] * len(empty_images), nrows=2, ncols=5)
plt.show()

