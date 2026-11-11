import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             roc_auc_score, precision_recall_curve)

def evaluasi_model_lengkap(model, X_train, y_train, X_test, y_test,
                           threshold=0.5, nama_model="Model", feature_names=None,
                           nama_file_simpan=None):

    print(f"\n" + "="*60)
    print(f"EVALUATION OF {nama_model.upper()} (THRESHOLD: {threshold})")
    print("="*60)

    # Predict probabilities
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Apply custom probability threshold
    y_pred_train_custom = (y_prob_train >= threshold).astype(int)
    y_pred_test_custom = (y_prob_test >= threshold).astype(int)

    # Compute training metrics
    acc_train = accuracy_score(y_train, y_pred_train_custom)
    prec_train = precision_score(y_train, y_pred_train_custom, zero_division=0)
    rec_train = recall_score(y_train, y_pred_train_custom, zero_division=0)
    f1_train = f1_score(y_train, y_pred_train_custom, zero_division=0)
    auc_train = roc_auc_score(y_train, y_prob_train)

    # Compute testing metrics
    acc_test = accuracy_score(y_test, y_pred_test_custom)
    prec_test = precision_score(y_test, y_pred_test_custom, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test_custom, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test_custom, zero_division=0)
    auc_test = roc_auc_score(y_test, y_prob_test)

    # Print comparative results table
    print(f"{'Metric':<12} | {'Training Data':<20} | {'Testing Data':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<12} | {acc_train:.4f}               | {acc_test:.4f}")
    print(f"{'Precision':<12} | {prec_train:.4f}               | {prec_test:.4f}")
    print(f"{'Recall':<12} | {rec_train:.4f}               | {rec_test:.4f}")
    print(f"{'F1-Score':<12} | {f1_train:.4f}               | {f1_test:.4f}")
    print(f"{'AUC-ROC':<12} | {auc_train:.4f}               | {auc_test:.4f}")
    print("-" * 60)

    # Visualization dashboard (4 subplots)
    plt.figure(figsize=(24, 5))

    # Plot 1: Confusion Matrix
    plt.subplot(1, 4, 1)
    cm = confusion_matrix(y_test, y_pred_test_custom)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Tsunami (0)', 'Tsunami (1)'],
                yticklabels=['Non-Tsunami (0)', 'Tsunami (1)'])
    plt.title(f'Confusion Matrix\n (Threshold = {threshold})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    # Plot 2: ROC Curve
    plt.subplot(1, 4, 2)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    plt.plot(fpr_train, tpr_train, label=f'Train AUC: {auc_train:.3f}')
    plt.plot(fpr_test, tpr_test, label=f'Test AUC: {auc_test:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Plot 3: Precision-Recall Curve
    plt.subplot(1, 4, 3)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob_test)
    plt.plot(rec_curve, prec_curve, marker='.', label='PR Curve')
    plt.axhline(y=prec_test, color='r', linestyle='--', label=f'Threshold {threshold}')
    plt.axvline(x=rec_test, color='r', linestyle='--')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Plot 4: Feature Importance
    plt.subplot(1, 4, 4)
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.best_estimator_.feature_importances_

    if feature_names is not None:
        imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        ax = sns.barplot(x=imp_series.values, y=imp_series.index, hue=imp_series.index, palette='viridis', legend=False)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')

        for p in ax.patches:
            bar_width = p.get_width()
            ax.annotate(f'{bar_width:.4f}',
                        (bar_width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        fontsize=10, color='black',
                        xytext=(5, 0), textcoords='offset points')
        plt.xlim(0, max(imp_series.values) * 1.15)

    plt.tight_layout()
    if nama_file_simpan is not None:
        plt.savefig(nama_file_simpan, dpi=600, bbox_inches='tight', pad_inches=0.1)
        print(f"[INFO] High-definition figure successfully saved: {nama_file_simpan}")
    plt.show()
