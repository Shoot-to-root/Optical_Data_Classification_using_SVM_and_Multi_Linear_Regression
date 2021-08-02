from sklearn.svm import SVC, NuSVC
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)

train = pd.read_csv("2class_trianing.csv", index_col=False)
test = pd.read_csv("2class_test.csv", index_col=False)
#train = pd.read_csv("4class_trianing.csv", index_col=False)
#test = pd.read_csv("4class_test.csv", index_col=False)
#print(train.head())
#print(train.dtypes)
train = train.fillna(train.mean())
test = test.fillna(test.mean())
#print(train.isnull().sum())
train = train.drop(["feature_0"], axis=1)
test = test.drop(["feature_0"], axis=1)
#print(train.head())

#split label
train_label = train.label
test_label = test.label
train = train.drop(["label"], axis=1)
test = test.drop(["label"], axis=1)
#print(train.shape, test.shape)
"""
#generate noise
mu, sigma = 5.0, 0.1 
feature_noise = np.random.normal(mu, sigma, [441,118])
#print(feature_noise.shape)
label_noise = np.random.rand(441)
#print(label_noise.shape)
noisy_train = train + feature_noise
noisy_label = train_label + label_noise
"""
"""
#tuning parameter
svc_param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'], 
                  'kernel': ['rbf', 'sigmoid']} 

nusvc_param_grid = {'nu': [0.01, 0.02, 0.03, 0.04, 0.05],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'], 
                    'kernel': ['rbf', 'sigmoid']}
 
svc = SVC()
nusvc = NuSVC()

clf = GridSearchCV(svc, svc_param_grid)
grid_result = clf.fit(train, train_label)
print(f"Best score: {grid_result.best_score_},\nBest params：{grid_result.best_params_}")
"""
#model = NuSVC(nu=0.1, gamma='scale', kernel='rbf')
model = SVC(C=10, gamma='scale', kernel='rbf')
#model.fit(noisy_train, noisy_label.astype('int'))
model.fit(train, train_label)
result = model.score(test, test_label)
pred = model.predict(test)
print(classification_report(test_label, pred))
print(f"Accuracy: {result}")
#result = pd.DataFrame(columns=["svm_score"], index=None)
#result.loc[0] = model.score(train, train_label)
#result.to_csv("svm_score.csv")

#plot learning curve
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

title = r"Learning Curves (SVM, RBF kernel, gamma=scale)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(model, title, train, train_label, axes=axes[:, 1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

plt.show()