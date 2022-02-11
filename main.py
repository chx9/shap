from fastapi import FastAPI
import base64
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
class Params(BaseModel):
    RASH: float
    NEUT: float
    HB: float
    AST: float
    DD: float
    ML: float
    SPL: float
    CD4: float


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})
@app.post('/explain')
async def explain(params: Params):
    params = dict(params)
    print(params)
    # load model
    excelFile = r'./训练集SHAP.xlsx'
    df = pd.DataFrame(pd.read_excel(excelFile))  # 读取表格，变量类型为dataframe
    x = df[['NEUT', 'RASH', 'CD4', 'AST', 'DD', 'SPL', 'HB']]  # 选取表格中特定列
    with open('./model.pkl', 'rb') as f:
        model = pickle.load(f)
    explainer = shap.Explainer(model, x, feature_names=['NEUT', 'RASH', 'CD4', 'AST', 'DD', 'SPL', 'HB'])


    arr = np.array([[params['NEUT'], params['RASH'], params['CD4'], params['AST'], params['DD'], params['SPL'], params['HB']]], dtype=np.float64)

    shap_values = explainer(arr)
    shap_plot = shap.plots.force(shap_values[0], matplotlib=True, show=False)
    buf = BytesIO()
    plt.savefig(buf,
                format="png",
                dpi=150,
                bbox_inches='tight')
    dataToTake = base64.b64encode(buf.getbuffer()).decode("ascii")

    return {"dataToTake": dataToTake}
