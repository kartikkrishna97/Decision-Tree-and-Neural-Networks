from utils import preprocess_data, get_one_hot_array
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train_path = "train.csv"
val_path = 'val.csv'
test_csv = 'test.csv'

X_train, Y_train, type, attributes = get_one_hot_array(train_path)
X_val, Y_val, type, attributes = get_one_hot_array(val_path)
X_test, Y_test, type, attributes = get_one_hot_array(val_path)

param_grid = {
    'n_estimators': [50, 150, 250],
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
    'min_samples_split': [2, 4, 6, 8]
}

rf = RandomForestClassifier(oob_score=True, random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, 
                           verbose=2, return_train_score=True)


grid_search.fit(X_train, Y_train)


print("Best parameters found: ", grid_search.best_params_)

# Print the best out of bag score 
print("Best out of bag score found: ", grid_search.best_estimator_.oob_score_)

test_accuracy = grid_search.best_estimator_.score(X_test, Y_test)
print("Test accuracy: ", test_accuracy)

val_accuracy = grid_search.best_estimator_.score(X_val, Y_val)
print("Val accuracy: ", val_accuracy)





