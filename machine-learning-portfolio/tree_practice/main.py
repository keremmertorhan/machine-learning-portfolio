from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load IRIS dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# 2. Split data into train and test sets (initial run with 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. Predict on train and test data
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# 5. Calculate accuracy
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# 6. Explanation
print("\nExplanation: The train accuracy tells us how well the model fits the training data.")
print("If it's too high compared to test accuracy, the model may be overfitting.")

# 7. Evaluate accuracy over 10 different random states
accuracies = []
for seed in range(10):
    X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(X, y, test_size=0.25, random_state=seed)
    model_rs = DecisionTreeClassifier()
    model_rs.fit(X_train_rs, y_train_rs)
    preds_rs = model_rs.predict(X_test_rs)
    acc = accuracy_score(y_test_rs, preds_rs)
    accuracies.append(acc)

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\n[Random State Testing - test_size=0.25]")
print(f"Average Accuracy over 10 runs: {mean_acc:.2f}")
print(f"Standard Deviation: {std_acc:.2f}")

# 8. Test different split ratios
split_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
split_accuracies = []

for ratio in split_ratios:
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=ratio, random_state=42)
    model_r = DecisionTreeClassifier()
    model_r.fit(X_train_r, y_train_r)
    preds_r = model_r.predict(X_test_r)
    acc = accuracy_score(y_test_r, preds_r)
    split_accuracies.append(acc)
    print(f"Test Size {ratio}: Accuracy = {acc:.2f}")

# 9. Plot accuracy vs test split ratio
plt.figure(figsize=(8, 5))
plt.plot(split_ratios, split_accuracies, marker='o', color='orange')
plt.title("Accuracy vs Test Set Size")
plt.xlabel("Test Set Ratio")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 10. Visualize the trained decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree Structure")
plt.show()
