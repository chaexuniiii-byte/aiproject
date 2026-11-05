import re
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="종로구 연령별 인구 분석",
    page_icon="💗",
    layout="wide"
)

HOT_PINK = "#FF1493"
AGE_GROUPS = [
    "0~9세", "10~19세", "20~29세", "30~39세",
    "40~49세", "50~59세", "60~69세", "70~79세",
    "80~89세", "90~99세", "100세 이상"
]
GENDER_LABELS = {"전체": "계", "남성": "남", "여성": "여"}
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]


# pages 폴더 안의 이 파일을 기준으로 프로젝트 최상위 폴더를 찾습니다.
# CSV는 app.py와 같은 최상위 폴더에 둡니다.
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "popuiation.csv"


@st.cache_data
def load_data(file_path):
    """한글 CSV 인코딩을 순서대로 시도해 데이터프레임을 반환합니다."""
    last_error = None

    for encoding in ("cp949", "utf-8-sig", "euc-kr"):
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise ValueError(
            "CSV 인코딩을 확인할 수 없습니다. "
            "CP949 또는 UTF-8 형식의 파일인지 확인해 주세요."
        ) from last_error

    if "행정구역" not in data.columns:
        raise ValueError("CSV에 '행정구역' 열이 없습니다.")

    for column in data.columns:
        if column != "행정구역":
            data[column] = pd.to_numeric(
                data[column].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    return data


def clean_district_name(value):
    name = str(value).strip()
    name = name.replace("서울특별시 종로구", "").strip()
    name = re.sub(r"\(\d+\)$", "", name).strip()
    return name if name else "종로구 전체"


# 반드시 위에서 df를 먼저 생성한 뒤 아래 분석 코드에서 사용합니다.
if not CSV_PATH.exists():
    st.error(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
    st.info(
        "popuiation.csv 파일을 GitHub 저장소의 최상위 폴더에 "
        "app.py와 함께 업로드해 주세요."
    )
    st.stop()

try:
    df = load_data(str(CSV_PATH))
except Exception as error:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()

df["행정구역_표시"] = df["행정구역"].apply(clean_district_name)

st.title("💗 종로구 연령별 인구 분석")
st.write("행정구역, 성별, 연도를 선택해 연령대별 인구수를 확인하세요.")

st.subheader("🔎 분석 조건 선택")
col1, col2, col3 = st.columns(3)

district_options = df["행정구역_표시"].tolist()

with col1:
    selected_district = st.selectbox(
        "📍 행정구역",
        district_options,
        index=0
