import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

# 1. Data Loading
xlsx1_filePath = "C:\\Users\\LL\\Desktop\\整体流程\\4.数据拼接\\拼接后的模板\\A.xlsx"
xlsx2_filePath = "C:\\Users\\LL\\Desktop\\整体流程\\4.数据拼接\\拼接后的模板\\B.xlsx"

data_1 = pd.read_excel(xlsx1_filePath)
data_2 = pd.read_excel(xlsx2_filePath)

# Merging datasets
data = pd.concat([data_1, data_2])

# Filling missing values and standardization
y = data['label'].reset_index(drop=True)
dataz = data.fillna(0)
X = dataz[dataz.columns[1:]]

colNames = X.columns
X = X.astype(np.float64)
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames

# 2. LASSO Feature Selection
alphas = np.logspace(-3, 1, 50)
model_lassoCV = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(X, y)
coef = pd.Series(model_lassoCV.coef_, index=X.columns)
index = coef[coef != 0].index
X = X[index]

# 3. Model Training and Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
sum_acc_model_mlp = []
sum_auc_model_mlp = []
sum_sen = []
sum_spe = []

y_true_all = []
y_pred_prob_all = []
auc_values = []


def create_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, 
                  loss='categorical_crossentropy', 
                  metrics=['acc'])
    return model

# Cross-validation
for train_index, test_index in kf.split(X):

    model = create_model()
    
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    #Training Model
    history = model.fit(
        X_train, pd.get_dummies(y_train),
        epochs=200, 
        batch_size=32, 
        verbose=0,
        validation_data=(X_test, pd.get_dummies(y_test)))
    
    # Evaluate Model
    score_mlp = model.evaluate(X_test, pd.get_dummies(y_test), verbose=0)
    sum_acc_model_mlp.append(score_mlp[1])
    
    y_pred_prob = model.predict(X_test)[:, 1]
    
    auc_values.append(roc_auc_score(y_test, y_pred_prob))
    
    # Summarize predictions and true labels
    y_true_all.extend(y_test)
    y_pred_prob_all.extend(y_pred_prob)
    
    report = classification_report(y_test, 
                                  np.argmax(model.predict(X_test), axis=1), 
                                  output_dict=True)
    sum_spe.append(report['0']['recall'])  
    sum_sen.append(report['1']['recall'])  

#Output the final result
print(f"Average accuracy rate: {np.mean(sum_acc_model_mlp):.4f} ± {np.std(sum_acc_model_mlp):.4f}")
print(f"AUC: {np.mean(auc_values):.4f} ± {np.std(auc_values):.4f}")
print(f"Average sensitivity: {np.mean(sum_sen):.4f} ± {np.std(sum_sen):.4f}")
print(f"Average specificity: {np.mean(sum_spe):.4f} ± {np.std(sum_spe):.4f}")

fpr, tpr, _ = roc_curve(y_true_all, y_pred_prob_all)
auc_all = roc_auc_score(y_true_all, y_pred_prob_all)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_all:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - All Data')
plt.legend(loc="lower right")
plt.show()