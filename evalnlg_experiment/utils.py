import scipy.stats
import random
import numpy as np
import catboost
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.utils.class_weight import compute_class_weight


def filt(lista, ind):
    return [lista[i] for i in range(len(lista)) if ind[i]==False]

def shuffle_list(input_list, random_state):
    output_list = input_list.copy()
    random.seed(random_state)
    random.shuffle(output_list)
    return output_list

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def calculate_correlation(x, y, correlation_type="pearson"):
    """
    Calculates the correlation between two lists x and y based on the specified type of correlation.
    
    Parameters:
    - x: list of numbers
    - y: list of numbers
    - correlation_type: string indicating the type of correlation (default is "pearson")
    
    Returns:
    - correlation value
    """
    
    if len(x) != len(y):
        raise ValueError("The two lists must have the same length!")
    
    if correlation_type.lower() == "pearson":
        return np.corrcoef(x, y)[0, 1]
    elif correlation_type.lower() == "spearman":
        return scipy.stats.spearmanr(x, y).correlation
    else:
        raise ValueError(f"Unsupported correlation type: {correlation_type}. Supported types are: pearson, spearman.")
 

 


def catb(X, y, test_size=0.1, balance=True, random_state=42):
    """
    Fit a CatBoost model using early stopping on a validation set.
    
    Parameters:
    - X: Features.
    - y: Target variable.
    - task: Either 'regression' or 'classification'.
    - test_size: Proportion of the dataset to include in the validation set.
    - random_state: Seed used by the random number generator for reproducibility.
    
    Returns:
    - model: Trained CatBoost model.
    """
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
  
    if balance:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
    else:
        class_weights = None
        
    # Initialize appropriate CatBoost model based on task
    model = catboost.CatBoostClassifier(iterations=5000,
                                        eval_metric='MultiClass',
                                        use_best_model=True, class_weights=class_weights)

    # Fit the model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
        early_stopping_rounds=500
    )
    
    return model