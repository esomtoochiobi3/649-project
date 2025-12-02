import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

# === load and clean data ===
df = pd.read_csv("features/features.csv")
for col in df.columns:
    if df[col].dtype == object and df[col].str.startswith("[").any():
        df[col] = df[col].apply(lambda x: float(x.strip("[]")))

X = df.drop(columns=["filename", "label"])
y = df["label"]
n_samples = len(y)

# === define classifiers ===
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM (Linear)": SVC(kernel="linear", C=1),
    "SVM (RBF)": SVC(kernel="rbf", C=1, gamma="scale"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

summary_rows = []
fold_score_rows = []

print("\n=== Cross-Validation Summary ===")
for name, model in models.items():
    fold_accuracies = []
    correct = 0
    incorrect = 0
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        correct += (y_pred == y_test).sum()
        incorrect += (y_pred != y_test).sum()

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    summary_rows.append({
        "Model": name,
        "Mean Accuracy": round(mean_acc, 4),
        "Std Dev": round(std_acc, 4),
        "Correct": correct,
        "Incorrect": incorrect,
        "Total Samples": n_samples
    })

    fold_score_rows.append({
        "Model": name,
        "Fold 1": fold_accuracies[0],
        "Fold 2": fold_accuracies[1],
        "Fold 3": fold_accuracies[2],
        "Fold 4": fold_accuracies[3],
        "Fold 5": fold_accuracies[4]
    })

    print(f"{name:15s} | Accuracy: {mean_acc:.4f} ± {std_acc:.4f} | Correct: {correct} / {n_samples}")

# === save tables ===
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/classifier_summary.csv", index=False)

fold_scores_df = pd.DataFrame(fold_score_rows).set_index("Model").T
fold_scores_df.to_csv("results/fold_scores.csv")

# === PCA visualization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Genre"] = y

plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Genre", palette="tab10", alpha=0.8)
plt.title("PCA of Audio Features by Genre")
plt.tight_layout()
plt.savefig("results/pca_plot.png", dpi=300)
plt.show()