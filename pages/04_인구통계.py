
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="종로구 인구 통계",
    page_icon="📊",
    layout="wide"
)

st.title("📊 종로구 동별 인구 통계")
st.caption("연도, 행정동, 성별에 따른 연령대별 인구를 확인해 보세요.")

# CSV 파일 위치
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "popuiation.csv"

# CSV 파일 읽기
@st.cache_data
def load_data(file_path):
    encodings = ["cp949", "utf-8-sig", "euc-kr"]

    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해 주세요.")

# 파일 확인 및 데이터 불러오기
if not CSV_PATH.exists():
    st.error("popuiation.csv 파일을 찾을 수 없습니다.")
    st.info("CSV 파일을 app.py와 같은 폴더에 넣어주세요.")
    st.stop()

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# 숫자 데이터 변환
for col in df.columns[1:]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

# 행정구역 이름 정리
df["행정구역"] = (
    df["행정구역"]
    .astype(str)
    .str.replace("서울특별시 종로구 ", "", regex=False)
    .str.replace(r"\s*\(\d+\)$", "", regex=True)
    .str.strip()
)

# 연도 및 성별 설정
years = ["2020", "2021", "2022", "2023", "2024", "2025"]
gender_options = {
    "전체": "계",
    "남성": "남",
    "여성": "여"
}

# 연령대
age_groups = [
    "0~9세",
    "10~19세",
    "20~29세",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상"
]

# 데이터에 있는 행정동 목록
districts = df["행정구역"].dropna().unique().tolist()

# 전체 행을 우선 선택
default_index = 0
if "종로구" in districts:
    default_index = districts.index("종로구")

st.subheader("🔎 조회 조건")

col1, col2, col3 = st.columns(3)

with col1:
    selected_district = st.selectbox(
        "행정구역 선택",
        districts,
        index=default_index
    )

with col2:
    selected_gender = st.selectbox(
        "성별 선택",
        list(gender_options.keys())
    )

with col3:
    selected_year = st.selectbox(
        "연도 선택",
        years,
        index=len(years) - 1
    )

# 선택한 연도와 성별에 해당하는 열 이름 만들기
gender_code = gender_options[selected_gender]

total_col = f"{selected_year}년_{gender_code}_계"
age_cols = [
    f"{selected_year}년_{gender_code}_{age}"
    for age in age_groups
]

# 실제 CSV 열 이름 확인
missing_cols = [
    col for col in [total_col] + age_cols
    if col not in df.columns
]

if missing_cols:
    st.error("CSV 파일에서 필요한 열을 찾지 못했습니다.")
    st.write("찾지 못한 열:", missing_cols)
    st.stop()

# 선택한 행정구역 데이터 찾기
selected_data = df[
    df["행정구역"] == selected_district
]

if selected_data.empty:
    st.warning("선택한 행정구역의 데이터가 없습니다.")
    st.stop()

selected_row = selected_data.iloc[0]

# 연령대별 인구 데이터
population = selected_row[age_cols].fillna(0).tolist()

# 열 이름에서 연령대만 표시
chart_labels = age_groups

# 전체 인구
total_population = selected_row[total_col]

# 주요 수치
st.subheader("👥 인구 현황")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "행정구역",
    selected_district
)

metric2.metric(
    "전체 인구",
    f"{total_population:,.0f}명"
    if pd.notna(total_population) else "자료 없음"
)

metric3.metric(
    "조회 기준",
    f"{selected_year}년 / {selected_gender}"
)

st.divider()

# 그래프
st.subheader("📈 연령대별 인구 분포")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_labels,
        y=population,
        mode="lines+markers",
        name="인구수",
        line=dict(
            color="#FF1493",
            width=4
        ),
        marker=dict(
            size=9,
            color="#FF1493"
        ),
        hovertemplate="%{x}<br>인구: %{y:,.0f}명<extra></extra>"
    )
)

fig.update_layout(
    xaxis_title="연령대",
    yaxis_title="인구수(명)",
    height=550,
    template="plotly_white",
    hovermode="x unified",
    font=dict(
        family="Malgun Gothic, sans-serif",
        size=13
    ),
    margin=dict(l=20, r=20, t=30, b=20)
)

fig.update_xaxes(
    type="category",
    categoryorder="array",
    categoryarray=age_groups,
    showgrid=False
)

fig.update_yaxes(
    rangemode="tozero",
    dtick=100,
    showgrid=True,
    gridcolor="#E5E5E5",
    tickformat=","
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# 데이터 표
with st.expander("📋 연령대별 인구 데이터 확인하기"):
    result_df = pd.DataFrame({
        "연령대": age_groups,
        "인구수": population
    })

    result_df["인구수"] = result_df["인구수"].fillna(0).astype(int)

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

st.caption("※ 자료는 제공된 popuiation.csv 파일을 기준으로 표시합니다.")
