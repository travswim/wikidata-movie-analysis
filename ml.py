# Standard Stuff
import sys
import pandas as pd
import numpy as np

# Tools
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
# Regressors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



def main(infile='ml.csv'):
    """
    Main ml pipeline
    """
    # Import data
    data = pd.read_csv(infile)
    data = data.drop(['audience_percent'], axis=1)
    data = data.drop(['audience_ratings'], axis=1)
    # Drop audience_average because they are bassically the same
    X = data.drop(['audience_average'], axis=1)
    y = data['audience_average']
    
    # TODO spilt the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # TODO Create some models
    # KNN
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=10, weights='distance')
    )

    # SVR
    svr_model = make_pipeline(
        StandardScaler(),
        SVR(gamma='scale', C=1.0, epsilon=0.2)
    )
    # Boosted
    boosted_model = make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor()
    )
    
    # Decision Tree
    tree_model = make_pipeline(
        StandardScaler(),
        DecisionTreeRegressor(random_state=0)
    )
    #  MLP
    mlp_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    )

    models = [knn_model, svr_model, boosted_model, tree_model, mlp_model]
    model_str = ['knn_model', 'svr_model', 'boosted_model', 'Decision Tree', 'mlp_model']
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        print(str(model_str[i]) + ": ")
        print(m.score(X_valid, y_valid))
    
    # TODO: Do a better analysis on the model that performs the best
    # Boosted performs the best so we're going to tune that
    boosted_pipeline = Pipeline([('scaler',  StandardScaler()),
            ('boosted', GradientBoostingRegressor())])
    boosted_params=[{'boosted__n_estimators':[100], \
        'boosted__loss': ['ls', 'lad', 'huber'],
        'boosted__learning_rate': [0.1, 0.05], \
            'boosted__max_depth':[4, 6], 'boosted__min_samples_leaf':[3,10],\
                'boosted__max_features':[1.0, 0.1]}]
    
    # Search the parameters
    clf = GridSearchCV(boosted_pipeline, boosted_params, cv=5, scoring='r2', verbose=10)
    print('trainning boosted')
    clf.fit(X_train, y_train)

    # Run the results
    print("Best Parameters: " + str(clf.best_params_))
    print("Training Score: " + str(clf.best_score_))
    print("Testing Score: " + str(clf.score(X_valid, y_valid)))

    # Best Parameters: {'boosted__learning_rate': 0.1, 'boosted__loss': 'ls', 'boosted__max_depth': 6, 'boosted__max_features': 0.1, 'boosted__min_samples_leaf': 3, 'boosted__n_estimators': 100}
    # Training Score: 0.5880016306361865
    # Testing Score: 0.5980893793534956
    
    # TODO: PCA
    # So we can do this:
    
    boosted_new_model = make_pipeline(
        PCA(17),
        StandardScaler(),
        GradientBoostingRegressor(learning_rate=0.1, \
            loss='ls', max_depth=6, max_features=0.1, \
                min_samples_leaf=3, n_estimators=100)
    )

    boosted_new_model.fit(X_train, y_train)
    print("Testing Score: " + str(boosted_new_model.score(X_valid, y_valid)))

    # PCA(17) Score: 0.57
    # No PCA  Score: 0.6
    # ?
if __name__ == '__main__':
    main()