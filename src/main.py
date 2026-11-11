from sklearn.model_selection import StratifiedKFold
from data_preprocessing import load_and_preprocess_data, prepare_datasets
from train_models import train_random_forest, train_xgboost
from evaluation import evaluasi_model_lengkap

if __name__ == "__main__":
    print("[INFO] Loading and preprocessing data...")
    path_gempa = "../data/raw/data2000-2025(csv).csv"
    path_tsunami = "../data/raw/katalog tsunami 2000-2025 japan.csv"
    
    df_processed = load_and_preprocess_data(path_gempa, path_tsunami)
    X_train, X_test, y_train, y_test, A_train, A_test, B_train, B_test = prepare_datasets(df_processed)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_params = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 4, 5, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5],
        'class_weight': ['balanced']
    }
    
    xgb_params_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    # 1. Random Forest - Dataset 1 (Without Fault Type)
    print("\n[INFO] Training Random Forest on Dataset 1...")
    best_rf_1 = train_random_forest(A_train, B_train, skf, rf_params)
    evaluasi_model_lengkap(best_rf_1, A_train, B_train, A_test, B_test, threshold=0.42, nama_model="Random Forest (Dataset 1)", feature_names=A_train.columns)

    # 2. Random Forest - Dataset 2 (With Fault Type)
    print("\n[INFO] Training Random Forest on Dataset 2...")
    best_rf_2 = train_random_forest(X_train, y_train, skf, rf_params)
    evaluasi_model_lengkap(best_rf_2, X_train, y_train, X_test, y_test, threshold=0.472, nama_model="Random Forest (Dataset 2)", feature_names=X_train.columns)

    # Calculate class imbalance ratios for XGBoost
    rasio_imbalance_1 = float(len(y_train[y_train==0]) / len(y_train[y_train==1]))
    rasio_imbalance_2 = float(len(B_train[B_train==0]) / len(B_train[B_train==1]))

    # 3. XGBoost - Dataset 1 (Without Fault Type)
    print("\n[INFO] Training XGBoost on Dataset 1...")
    best_xgb_1 = train_xgboost(A_train, B_train, skf, xgb_params_grid, rasio_imbalance_2)
    evaluasi_model_lengkap(best_xgb_1, A_train, B_train, A_test, B_test, threshold=0.57, nama_model="XGBoost (Dataset 1)", feature_names=A_train.columns)

    # 4. XGBoost - Dataset 2 (With Fault Type)
    print("\n[INFO] Training XGBoost on Dataset 2...")
    best_xgb_2 = train_xgboost(X_train, y_train, skf, xgb_params_grid, rasio_imbalance_1)
    evaluasi_model_lengkap(best_xgb_2, X_train, y_train, X_test, y_test, threshold=0.5, nama_model="XGBoost (Dataset 2)", feature_names=X_train.columns, nama_file_simpan="../reports/figures/grafik_evaluasi_xgboost_hd.png")
