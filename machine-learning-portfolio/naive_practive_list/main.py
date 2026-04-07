from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2. Save dataset info to log file
with open("log.txt", "w") as f:
    f.write(f"Shape: {X.shape}\n")
    f.write(f"Unique labels: {np.unique(y)}\n")

# 3. Access .data and .target directly (already done above)

# 4. Display a few samples
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
plt.suptitle("Sample MNIST Digits")
plt.show()

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 6. Train Naive Bayes model (MultinomialNB is best for positive pixel data)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 8. Accuracy and error analysis
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Error per class
error_per_class = 1 - cm.diagonal() / cm.sum(axis=1)
print("\nError rate per class:")
for i, rate in enumerate(error_per_class):
    print(f"Digit {i}: {rate:.2%}")
