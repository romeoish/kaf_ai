import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Type
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
import xgboost as xgb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

##### ===================== KS-2 연신&방사 feature를 활용한 target 값 예측 =====================


class FeatureInput_KS_2_Y(BaseModel):
# -------------- KS-2 방사 feature -----------------
    F_Roll_속도: float = Field(..., alias="F/Roll 속도")
    EXT_1호기_4_6: float = Field(..., alias="EXT 1호기 #4~#6")
    EXT_2호기_4_6: float = Field(..., alias="EXT 2호기 #4~#6")
    EXT_3호기_4_6: float = Field(..., alias="EXT 3호기 #4~#6")
    Dowtherm_온도: float = Field(..., alias="Dowtherm 온도")
    SP_속도: float = Field(..., alias="S/P 속도")
    방사_속도: float = Field(..., alias="방사 속도")
    Stacker_Sun_간격: float = Field(..., alias="Stacker Sun' 간격")
    QA_온도: float = Field(..., alias="Q/A 온도")
    QA_습도: float = Field(..., alias="Q/A 습도")
    QA_풍압_1: float = Field(..., alias="Q/A 풍압-1")
    QA_풍압_2: float = Field(..., alias="Q/A 풍압-2")
    QA_풍압_3: float = Field(..., alias="Q/A 풍압-3")
    QA_배기_1: float = Field(..., alias="Q/A 배기-1")
    QA_배기_2: float = Field(..., alias="Q/A 배기-2")
    QA_배기_3: float = Field(..., alias="Q/A 배기-3")    
# ------------- KS-2 연신 feature--------------
    CAN_수: float = Field(..., alias="CAN 수")
    CR_Box_압력: float = Field(..., alias="CR Box 압력")
    CR_Roll_압력: float = Field(..., alias="CR Roll 압력")
    CR_속도: float = Field(..., alias="CR 속도")
    Cutter_속도: float = Field(..., alias="Cutter 속도")
    DS_1_연신비: float = Field(..., alias="DS-1 연신비")
    DS_2_속도: float = Field(..., alias="DS-2 속도")
    Dryer_Zone_1: float = Field(..., alias="Dryer Zone #1")
    Dryer_Zone_2: float = Field(..., alias="Dryer Zone #2") 
    Residence: float
    Steam_압력: float = Field(..., alias="Steam 압력")
    TGS_속도비: float = Field(..., alias="TGS 속도비")
    TTS_속도: float = Field(..., alias="TTS 속도")
    길이: float
    분사량: float
    HAC_온도_상: float = Field(..., alias="HAC 온도_상")
    HAC_온도_하: float = Field(..., alias="HAC 온도_하")

    class Config:
        allow_population_by_field_name = True

def safe_split_column(value, index, default_value=0):
    try:
        if pd.isna(value) or value == " " or value == "":
            return default_value
        parts = str(value).split('/')
        if index < len(parts):
            return int(parts[index].strip())
        else:
            return default_value
    except:
        return default_value
    
def split_slash_columns(df):
    import re
    df_expanded = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].astype(str).str.contains('/').any():
            sample = df[col].dropna().astype(str).iloc[0]
            if re.match(r'^\s*\d+\.?\d*\s*/\s*\d+\.?\d*\s*$', sample):
                base = col.replace('(상/하)', '').replace('상/하', '').strip()
                col1, col2 = f"{base}_상", f"{base}_하"
                df_expanded[[col1, col2]] = df[col].str.split('/', expand=True).astype(float)
                df_expanded.drop(columns=[col], inplace=True)
    return df_expanded

def predict_KS_2_Y(features: FeatureInput_KS_2_Y):
    path = "KS_2_Y"
    os.makedirs(path, exist_ok=True)

    model_file = f"{path}/model_multi.joblib"
    feature_cols_file = f"{path}/feature_cols.joblib"
    model_file_denial = f"{path}/denial_model.joblib"
    feature_cols_file_denial = f"{path}/denial_feature_cols.joblib"

    model_multi = joblib.load(model_file)
    feature_cols = joblib.load(feature_cols_file)
    model_denial = joblib.load(model_file_denial)
    feature_cols_denial = joblib.load(feature_cols_file_denial)

    input_df = pd.DataFrame([features.dict(by_alias=True)])
    input_df = input_df[feature_cols]
    input_df_denial = input_df[feature_cols_denial]

    pred = model_multi.predict(input_df)
    pred_denial = model_denial.predict(input_df_denial)

    return {
        "result": {
            "Denier": float(pred_denial[0][0]),
            "Tenacity": float(pred[0][2]),
            "Elongation": float(pred[0][3]),
            "Total Finish": float(pred[0][0])
        }
    }

## uvicorn fastapi ##

# CORS 설정
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(middleware=middleware)

# 예측 엔드포인트 (KS-2 연신 전용)
@app.post("/predict_KS_2_Y")
async def predict_target(features: FeatureInput_KS_2_Y):  # ✅ Pydantic 모델로 명시
    try:
        result = predict_KS_2_Y(features=features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 오류: {str(e)}")

# 로컬 서버 구동
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)