import pandas as pd

# Load wine quality dataset
df = pd.read_csv("winequality-red.csv")

print("Dataset Shape:", df.shape)
df.head()

# 3. Convert To Binary
#A=0 (<5)
#B=1 (>=5)
df['quality'] = df['quality'].apply(lambda x: 0 if x < 5 else 1)

print(df['quality'].value_counts())

pd.unique(df['quality'])

df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# Plot Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# pi chart (class distribution)
import matplotlib.pyplot as plt

df['quality'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Class Distribution (A=0, B=1)")
plt.ylabel("")
plt.show()

X = df.drop("quality", axis=1)
y = df["quality"]

# Train and split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train

# Build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

cm = confusion_matrix(y_test, y_pred)
tp = cm[0,0]
tn = cm[1,1]
fp = cm[1,0] # false Positive
fn = cm[0,1] # False Negative

accuaracy = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall = tp / (tp + fn)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro') # harmoic mean of recall and precision 

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Build Decision Tree Classifier with hyperparameters

from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    splitter='best',
    min_samples_split=10,
    min_samples_leaf=2
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#parameter tuning

accuracy_dict = dict()
for depth in range(1, 11):
    for split in range(2,10):
        for leaf in range(1,5):
            clf = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[(depth, split, leaf)] = accuracy
            print(f"Depth: {depth}, Split: {split}, Leaf: {leaf} => Accuracy: {accuracy:.4f}")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time

param_grid = {
    'max_depth': [3,5,10,None],
    'min_samples_split': [2,5,10],
    'criterion': ['gini','entropy']
}

start = time.time()

grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_dt.fit(X_train, y_train)

end = time.time()

best_dt = grid_dt.best_estimator_

y_pred = best_dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred)

print("Best Parameters:", grid_dt.best_params_)
print("Accuracy:", acc_dt)
print("Time Taken:", end-start)


# Decision Tree diagram
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

plot_tree(best_dt,
          feature_names=X.columns,
          class_names=["A (<5)", "B (>=5)"],
          filled=True,
          rounded=True)

plt.title("Decision Tree Structure")
plt.show()