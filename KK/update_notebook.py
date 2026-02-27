import json

def create_code_cell(source_code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source_code.split('\n')[:-1]] + [source_code.split('\n')[-1]]
    }

def create_markdown_cell(source_code):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source_code.split('\n')[:-1]] + [source_code.split('\n')[-1]]
    }

with open('Decision_tree.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fixes
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'sns.heatmap(data.corr()' in source:
            cell['source'] = [line.replace('data.corr()', 'df.corr()') for line in cell['source']]
        if len(cell['source']) > 0 and cell['source'][0].startswith('cm = confusion_matrix(y_test, y_pred)'):
            cell['source'] = ['from sklearn.metrics import confusion_matrix\n', 'y_pred = clf.predict(X_test)\n'] + cell['source']

# New Cells
markdown_1 = "## Random Forest and Decision Tree Comparison (Grid, Random, Halving)"
code_1 = """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint

# Dictionary to store results
results = []
"""

code_2 = """# Decision Tree Hyperparameter Space
dt_param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

dt_param_dist = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 11),
    'criterion': ['gini', 'entropy']
}

# 1. Grid Search for Decision Tree
start_time = time.time()
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, n_jobs=-1)
dt_grid.fit(X_train, y_train)
dt_grid_time = time.time() - start_time
dt_grid_acc = accuracy_score(y_test, dt_grid.predict(X_test))
results.append({'Model': 'Decision Tree', 'Search Strategy': 'Grid Search', 'Accuracy': dt_grid_acc, 'Time (s)': dt_grid_time, 'Best Params': str(dt_grid.best_params_)})

# 2. Random Search for Decision Tree
start_time = time.time()
dt_random = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), dt_param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
dt_random.fit(X_train, y_train)
dt_random_time = time.time() - start_time
dt_random_acc = accuracy_score(y_test, dt_random.predict(X_test))
results.append({'Model': 'Decision Tree', 'Search Strategy': 'Random Search', 'Accuracy': dt_random_acc, 'Time (s)': dt_random_time, 'Best Params': str(dt_random.best_params_)})

# 3. Successive Halving for Decision Tree
start_time = time.time()
dt_halving = HalvingGridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, factor=2, random_state=42, n_jobs=-1)
dt_halving.fit(X_train, y_train)
dt_halving_time = time.time() - start_time
dt_halving_acc = accuracy_score(y_test, dt_halving.predict(X_test))
results.append({'Model': 'Decision Tree', 'Search Strategy': 'Successive Halving', 'Accuracy': dt_halving_acc, 'Time (s)': dt_halving_time, 'Best Params': str(dt_halving.best_params_)})

print("Decision Tree tuning completed.")
"""

code_3 = """# Random Forest Hyperparameter Space
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf_param_dist = {
    'n_estimators': randint(50, 150),
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 11)
}

# 1. Grid Search for Random Forest
start_time = time.time()
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_grid_time = time.time() - start_time
rf_grid_acc = accuracy_score(y_test, rf_grid.predict(X_test))
results.append({'Model': 'Random Forest', 'Search Strategy': 'Grid Search', 'Accuracy': rf_grid_acc, 'Time (s)': rf_grid_time, 'Best Params': str(rf_grid.best_params_)})

# 2. Random Search for Random Forest
start_time = time.time()
rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_dist, n_iter=10, cv=5, random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
rf_random_time = time.time() - start_time
rf_random_acc = accuracy_score(y_test, rf_random.predict(X_test))
results.append({'Model': 'Random Forest', 'Search Strategy': 'Random Search', 'Accuracy': rf_random_acc, 'Time (s)': rf_random_time, 'Best Params': str(rf_random.best_params_)})

# 3. Successive Halving for Random Forest
start_time = time.time()
rf_halving = HalvingGridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, factor=2, random_state=42, n_jobs=-1)
rf_halving.fit(X_train, y_train)
rf_halving_time = time.time() - start_time
rf_halving_acc = accuracy_score(y_test, rf_halving.predict(X_test))
results.append({'Model': 'Random Forest', 'Search Strategy': 'Successive Halving', 'Accuracy': rf_halving_acc, 'Time (s)': rf_halving_time, 'Best Params': str(rf_halving.best_params_)})

print("Random Forest tuning completed.")
"""

code_4 = """# Convert results to DataFrame
results_df = pd.DataFrame(results)
display(results_df)

# Heatmap Comparison of Accuracy
heatmap_data = results_df.pivot(index='Model', columns='Search Strategy', values='Accuracy')
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.4f', cbar_kws={'label': 'Accuracy'})
plt.title('Accuracy Heatmap: Models vs Search Strategies', fontsize=16)
plt.tight_layout()
plt.show()

# Heatmap Comparison of Time
heatmap_time = results_df.pivot(index='Model', columns='Search Strategy', values='Time (s)')
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_time, annot=True, cmap='Reds', fmt='.2f', cbar_kws={'label': 'Time (seconds)'})
plt.title('Time Efficiency Heatmap: Models vs Search Strategies', fontsize=16)
plt.tight_layout()
plt.show()

# Performance vs Efficiency Plot (Accuracy vs Time)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Time (s)', y='Accuracy', hue='Model', style='Search Strategy', s=200, palette='Set1')
plt.title('Efficiency vs Performance: Hyperparameter Search Strategies', fontsize=16)
plt.xlabel('Time Taken (seconds)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""

nb['cells'].append(create_markdown_cell(markdown_1))
nb['cells'].append(create_code_cell(code_1))
nb['cells'].append(create_code_cell(code_2))
nb['cells'].append(create_code_cell(code_3))
nb['cells'].append(create_code_cell(code_4))

with open('Decision_tree.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
