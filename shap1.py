import base64
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# load model
excelFile = r'.\训练集SHAP.xlsx'
df = pd.DataFrame(pd.read_excel(excelFile))  # 读取表格，变量类型为dataframe
X = df[['NEUT', 'RASH', 'CD4', 'AST', 'DD', 'SPL', 'HB']]  # 选取表格中特定列
with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)
explainer = shap.Explainer(model, X, feature_names=['NEUT', 'RASH', 'CD4', 'AST', 'DD', 'SPL', 'HB'])

file = r'.\验证集SHAP.xlsx'
RASH, NEUT, HB, AST, DD, ML, SPL, CD4 = 1, 2.5, 60, 50, 200, 1, 1, 10

arr = np.array([[NEUT, RASH, CD4, AST, DD, SPL, HB]], dtype=np.float64)


shap_values = explainer(arr)
shap_plot = shap.plots.force(shap_values[0], matplotlib=True, show=True)
exit()
buf = BytesIO()
plt.savefig(buf,
            format="png",
            dpi=150,
            bbox_inches='tight')
dataToTake = base64.b64encode(buf.getbuffer()).decode("ascii")
exit()
