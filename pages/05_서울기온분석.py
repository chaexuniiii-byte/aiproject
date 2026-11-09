
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# =====================================
# 1. 화면 설정
# =====================================
st.set_page_config(
    page_title="서울 기온 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 분석")
st.write("서울의 기온 데이터를 다양한 그래프로 분석해 보세요.")

# =====================================
# 2. CSV 파일 불러오기
# =====================================
# pages 폴더의 상위 폴더에서 CSV 찾기
CSV_PATH = Path(__file__).resolve().parent.parent / "temperature.csv"

if not CSV_PATH.exists():
    st.error("temperature.csv 파일을 찾을 수 없습니다.")
    st.stop()


def read_csv_file(path):
    for encoding in ["cp949", "utf-8-sig", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 인코딩을 확인해 주세요.")


try:
    df = read_csv_file(CSV_PATH)
except Exception as e:
    st.error(f"CSV 파일 오류: {e}")
    st.stop()

# =====================================
# 3. 필요한 열 확인
# =====================================
required_columns = [
    "일시",
    "평균기온(℃)",
    "최고기온(℃)",
    "최저기온(℃)"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"필요한 열이 없습니다: {missing}")
    st.write("현재 CSV 열 이름:", df.columns.tolist())
    st.stop()

# =====================================
# 4. 데이터 전처리
# =====================================
# 예: Oct-07 형식 날짜 변환
date_text = df["일시"].astype(str).str.strip()

df["일시"] = pd.to_datetime(
    date_text,
    format="%b-%y",
    errors="coerce"
)

# 다른 날짜 형식에 대한 보조 처리
invalid = df["일시"].isna()

if invalid.any():
    df.loc[invalid, "일시"] = pd.to_datetime(
        date_text.loc[invalid],
        errors="coerce"
    )

temperature_columns = [
    "평균기온(℃)",
    "최고기온(℃)",
    "최저기온(℃)"
]

for col in temperature_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=["일시"] + temperature_columns
).copy()

if df.empty:
    st.error("분석할 수 있는 데이터가 없습니다.")
    st.stop()

df["연도"] = df["일시"].dt.year
df["월"] = df["일시"].dt.month

# =====================================
# 5. 한글 폰트 설정
# =====================================
available_fonts = {
    font.name for font in fm.fontManager.ttflist
}

font_candidates = [
    "Malgun Gothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR"
]

for font in font_candidates:
    if font in available_fonts:
        plt.rcParams["font.family"] = font
        break

plt.rcParams["axes.unicode_minus"] = False

# =====================================
# 6. 분석 메뉴
# =====================================
st.divider()

analysis_type = st.radio(
    "📊 원하는 분석을 선택하세요.",
    [
        "① 서울의 월별 평균기온",
        "② 연도별 최고기온과 최저기온",
        "③ 두 연도 비교 분석"
    ],
    horizontal=False
)

# =====================================
# 기능 1. 월별 평균기온
# =====================================
if analysis_type == "① 서울의 월별 평균기온":

    st.subheader("🌸 서울의 월별 평균기온")

    monthly = (
        df.groupby("월")["평균기온(℃)"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        monthly.index,
        monthly.values,
        color="#FF5A00",
        marker="o",
        linewidth=3,
        markersize=8,
        label="월별 평균기온"
    )

    ax.set_title(
        "서울의 월별 평균기온",
        fontsize=20,
        pad=20
    )

    ax.set_xlabel("월", fontsize=13)
    ax.set_ylabel("평균기온 (℃)", fontsize=13)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [f"{i}월" for i in range(1, 13)]
    )

    ax.legend()
    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("📋 월별 평균기온 데이터")

    result = monthly.reset_index()
    result.columns = ["월", "평균기온(℃)"]
    result["평균기온(℃)"] = result[
        "평균기온(℃)"
    ].round(2)

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    # 가장 더운 달과 추운 달
    if monthly.notna().any():
        hottest = monthly.idxmax()
        coldest = monthly.idxmin()

        col1, col2 = st.columns(2)

        col1.metric(
            "🔥 가장 더운 달",
            f"{hottest}월",
            f"{monthly.max():.1f}℃"
        )

        col2.metric(
            "❄️ 가장 추운 달",
            f"{coldest}월",
            f"{monthly.min():.1f}℃"
        )

# =====================================
# 기능 2. 연도별 최고기온과 최저기온
# =====================================
elif analysis_type == "② 연도별 최고기온과 최저기온":

    st.subheader("📈 연도별 최고기온과 최저기온")

    yearly = (
        df.groupby("연도")
        .agg({
            "최고기온(℃)": "max",
            "최저기온(℃)": "min"
        })
        .reset_index()
        .sort_values("연도")
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # 최고기온: 형광 주황
    ax.plot(
        yearly["연도"],
        yearly["최고기온(℃)"],
        color="#FF5A00",
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="최고기온"
    )

    # 최저기온: 형광 파랑
    ax.plot(
        yearly["연도"],
        yearly["최저기온(℃)"],
        color="#00BFFF",
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="최저기온"
    )

    ax.set_title(
        "연도별 최고기온과 최저기온",
        fontsize=20,
        pad=20
    )

    ax.set_xlabel("연도", fontsize=13)
    ax.set_ylabel("온도 (℃)", fontsize=13)

    ax.legend(fontsize=12)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    years = yearly["연도"].tolist()

    if len(years) > 20:
        step = max(1, len(years) // 15)
        ax.set_xticks(years[::step])

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("📋 연도별 기온 데이터")

    st.dataframe(
        yearly.round(1),
        use_container_width=True,
        hide_index=True
    )

# =====================================
# 기능 3. 두 연도 비교
# =====================================
elif analysis_type == "③ 두 연도 비교 분석":

    st.subheader("📊 두 연도 월별 평균기온 비교")

    years = sorted(
        df["연도"].dropna().unique().tolist()
    )

    if len(years) < 2:
        st.warning("비교할 연도가 2개 이상 필요합니다.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        year1 = st.selectbox(
            "첫 번째 연도 선택",
            years,
            index=0
        )

    with col2:
        year2_options = [
            y for y in years if y != year1
        ]

        year2 = st.selectbox(
            "두 번째 연도 선택",
            year2_options,
            index=0
        )

    # 연도별 월평균 기온 계산
    comparison = (
        df[df["연도"].isin([year1, year2])]
        .groupby(["연도", "월"])["평균기온(℃)"]
        .mean()
        .unstack("연도")
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # 첫 번째 연도
    ax.plot(
        range(1, 13),
        comparison[year1],
        color="#FF5A00",
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=f"{year1}년"
    )

    # 두 번째 연도
    ax.plot(
        range(1, 13),
        comparison[year2],
        color="#00BFFF",
        marker="o",
        linewidth=2.5,
        markersize=7,
        label=f"{year2}년"
    )

    ax.set_title(
        f"{year1}년과 {year2}년 월별 평균기온 비교",
        fontsize=18,
        pad=20
    )

    ax.set_xlabel("월", fontsize=13)
    ax.set_ylabel("평균기온 (℃)", fontsize=13)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        [f"{i}월" for i in range(1, 13)]
    )

    ax.legend(fontsize=12)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 비교 데이터 표
    st.subheader("📋 두 연도 기온 비교 데이터")

    comparison_table = comparison.copy()
    comparison_table.columns = [
        f"{year1}년 평균기온",
        f"{year2}년 평균기온"
    ]

    comparison_table.index = [
        f"{m}월" for m in range(1, 13)
    ]

    st.dataframe(
        comparison_table.round(1),
        use_container_width=True
    )

    # 두 연도 평균기온 차이
    difference = (
        comparison[year1] - comparison[year2]
    ).mean()

    st.metric(
        "두 연도의 연평균기온 차이",
        f"{abs(difference):.2f}℃",
        delta=(
            f"{year1}년이 더 높음"
            if difference > 0
            else f"{year2}년이 더 높음"
            if difference < 0
            else "차이 없음"
        )
    )

# =====================================
# 7. 하단 안내
# =====================================
st.divider()

st.caption(
    "🌡️ 서울 기온 데이터 분석 프로그램 | "
    "자료에 포함된 값만을 이용하여 분석합니다."
)
