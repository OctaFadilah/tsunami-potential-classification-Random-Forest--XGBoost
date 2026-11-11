from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train, skf, rf_params):
    # Initialize Random Forest with class weighting to handle data imbalance
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_search = GridSearchCV(rf_base, rf_params, scoring='recall', cv=skf, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    return rf_search.best_estimator_

def train_xgboost(X_train, y_train, skf, xgb_params_grid, scale_pos_weight_val):
    # Initialize XGBoost with scale_pos_weight for class imbalance handling
    xgb_base = XGBClassifier(scale_pos_weight=scale_pos_weight_val, random_state=42, eval_metric='logloss')
    xgb_search = GridSearchCV(estimator=xgb_base, param_grid=xgb_params_grid, scoring='recall', cv=skf, n_jobs=-1, error_score='raise')
    xgb_search.fit(X_train, y_train)
    return xgb_search.best_estimator_
