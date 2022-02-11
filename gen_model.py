import pickle

import sklearn
import numpy as np
import pandas as pd
import shap
excelFile = r'./训练集SHAP(2.11)(1).xlsx'
df = pd.DataFrame(pd.read_excel(excelFile))#读取表格，变量类型为dataframe
X= df[['NEUT', 'RASH', 'CD4', 'AST', 'DD', 'SPL','HB']]#选取表格中特定列
y = df[['GROUP']]#y1是dataframe类型

model = sklearn.linear_model.LogisticRegression(penalty='none', C=1,solver= 'newton-cg')
model.fit(X, y.values.ravel())
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
