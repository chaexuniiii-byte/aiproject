
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    # pages 폴더의 상위 폴더에서 CSV 파일 찾기
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "Iris.csv"

    if not csv_path.exists():
        st.error(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        st.stop()

    # CP949 인코딩으로 CSV 읽기
    df = pd.read_csv(csv_path, encoding="cp949")

    # 열 이름 앞뒤 공백 제거
    df.columns = df.columns.str.strip()

    # 기존 한글 열 이름을 영어로 변경
    df = df.rename(columns={
        "꽃받침 길이": "sepal_length",
        "꽃받침 너비": "sepal_width",
        "꽃잎 길이": "petal_length",
        "꽃잎 너비": "petal_width",
        "품종": "species"
    })

    return df
