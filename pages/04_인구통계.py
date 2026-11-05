
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

GENDER_LABELS = {
    "전체": "계",
    "남성": "남",
    "여성": "여"
}

YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]

# CSV 파일 위치
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "popuiation.csv"


# CSV 불러오기
@st.cache_data
def load_data(file_path):
    for encoding in ["cp949", "utf-8-sig", "euc-kr"]:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV 한글 인코딩을 확인해 주세요.")

    # 열 이름의 공백과 BOM 제거
    data.columns = [
        str(col).replace("\ufeff", "").strip()
        for col in data.columns
    ]

    if "행정구역" not in data.columns:
        raise ValueError(
            "CSV에서 '행정구역' 열을 찾을 수 없습니다."
        )

    # 인구 데이터 숫자로 변환
    for col in data.columns:
        if col != "행정구역":
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    return data


# 파일 확인
if not CSV_PATH.exists():
    st.error("popuiation.csv 파일을 찾을 수 없습니다.")
    st.info("CSV 파일을 app.py와 같은 폴더에 넣어주세요.")
    st.stop()

try:
    df = load_data(str(CSV_PATH))
except Exception as e:
    st.error(f"CSV 파일 오류: {e}")
    st.stop()


# 행정구역 이름 정리
def clean_district_name(value):
    name = str(value).strip()
    name = name.replace("서울특별시 종로구", "").strip()
    name = re.sub(r"\(\d+\)$", "", name).strip()

    return name if name else "종로구 전체"


df["행정구역_표시"] = df["행정구역"].apply(clean_district_name)


# 제목
st.title("💗 종로구 연령별 인구 분석")
st.write(
    "행정구역, 성별, 연도를 선택해 "
    "연령대별 인구 분포를 확인해 보세요."
)


# 선택 메뉴
st.subheader("🔎 분석 조건 선택")

col1, col2, col3 = st.columns(3)

district_options = df["행정구역_표시"].tolist()

with col1:
    selected_district = st.selectbox(
        "📍 행정구역",
        district_options,
        index=0
    )

with col2:
    selected_gender = st.selectbox(
        "👥 성별",
        list(GENDER_LABELS.keys())
    )

with col3:
    selected_year = st.selectbox(
        "📅 연도",
        YEARS,
        index=len(YEARS) - 1
    )


# 선택한 행정구역 찾기
matched_rows = df[
    df["행정구역_표시"] == selected_district
]

if matched_rows.empty:
    st.error("선택한 행정구역의 데이터가 없습니다.")
    st.stop()

selected_row = matched_rows.iloc[0]


# 실제 CSV 열 이름을 기준으로 연령대 열 찾기
gender_code = GENDER_LABELS[selected_gender]
prefix = f"{selected_year}년_{gender_code}_"

# 공백을 제거한 열 이름으로 비교
column_lookup = {
    re.sub(r"\s+", "", str(col)): col
    for col in df.columns
}

age_columns = []

for age in AGE_GROUPS:
    target = re.sub(r"\s+", "", prefix + age)

    if target in column_lookup:
        age_columns.append(column_lookup[target])
    else:
        age_columns.append(None)


# 찾지 못한 열 확인
missing_ages = [
    AGE_GROUPS[i]
    for i, col in enumerate(age_columns)
    if col is None
]

if missing_ages:
    st.error("선택한 조건의 연령대 열을 찾지 못했습니다.")

    st.write("찾지 못한 연령대:", missing_ages)

    st.write("CSV에 실제로 들어 있는 해당 연도 열 이름:")
    actual_columns = [
        str(col)
        for col in df.columns
        if str(col).startswith(prefix)
    ]
    st.code("\n".join(actual_columns) if actual_columns else "해당 열 없음")

    st.stop()


# 연령대별 인구 데이터
population_values = []

for col in age_columns:
    value = selected_row[col]

    if pd.isna(value):
        value = 0

    population_values.append(int(value))


# 연령대별 인구 합계
# 별도의 전체 인구 열 이름에 의존하지 않습니다.
total_population = sum(population_values)


# 인구 현황
st.divider()
st.subheader("👥 인구 현황")

m1, m2, m3 = st.columns(3)

m1.metric("선택한 행정구역", selected_district)
m2.metric("선택한 연도", f"{selected_year}년")
m3.metric("연령대별 인구 합계", f"{total_population:,}명")


# 그래프
st.divider()
st.subheader("📈 연령대별 인구수")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=AGE_GROUPS,
        y=population_values,
        mode="lines+markers",
        name="인구수",
        line=dict(
            color=HOT_PINK,
            width=4
        ),
        marker=dict(
            color=HOT_PINK,
            size=9
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "인구수: %{y:,.0f}명"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title=f"{selected_district} · {selected_gender} · {selected_year}년",
    xaxis=dict(
        title="연령대",
        type="category",
        categoryorder="array",
        categoryarray=AGE_GROUPS,
        showgrid=True,
        gridcolor="#F3B6D2"
    ),
    yaxis=dict(
        title="인구수(명)",
        rangemode="tozero",
        dtick=100,
        tickformat=",d",
        showgrid=True,
        gridcolor="#E8DCE2"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(
        family="Arial, sans-serif",
        size=13,
        color="#333333"
    ),
    hovermode="x unified",
    height=540
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# 데이터 표
with st.expander("📋 그래프에 사용된 인구 데이터 보기"):
    result_df = pd.DataFrame({
        "연령대": AGE_GROUPS,
        "인구수": population_values
    })

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "자료: popuiation.csv | 연령대는 10세 단위입니다."
)
