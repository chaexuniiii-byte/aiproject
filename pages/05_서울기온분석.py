
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# -----------------------------
# 1. 화면 설정
# -----------------------------
st.set_page_config(
    page_title="월별 기온분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 기온분석")
st.write("원하는 월을 선택하면 연도별 최고기온과 최저기온을 비교할 수 있습니다.")

# -----------------------------
# 2. CSV 파일 읽기
# -----------------------------
def read_csv_file(file):
    for encoding in ["cp949", "utf-8-sig", "euc-kr", "utf-8"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 인코딩을 확인해 주세요.")

uploaded_file = st.file_uploader(
    "기온 데이터 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is not None:
    source_file = uploaded_file
else:
    default_file = Path(__file__).parent / "temperature.csv"

    if default_file.exists():
        source_file = default_file
        st.info("temperature.csv 파일을 불러왔습니다.")
    else:
        st.warning("CSV 파일을 업로드해 주세요.")
        st.stop()

try:
    df = read_csv_file(source_file)
except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

# -----------------------------
# 3. 필요한 열 확인
# -----------------------------
required_columns = [
    "일시",
    "최고기온(℃)",
    "최저기온(℃)"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"필요한 열이 없습니다: {missing}")
    st.write("현재 열 이름:", df.columns.tolist())
    st.stop()

# -----------------------------
# 4. 날짜 및 기온 데이터 처리
# -----------------------------
df["일시"] = pd.to_datetime(
    df["일시"].astype(str).str.strip(),
    format="%b-%y",
    errors="coerce"
)

df["최고기온(℃)"] = pd.to_numeric(
    df["최고기온(℃)"],
    errors="coerce"
)

df["최저기온(℃)"] = pd.to_numeric(
    df["최저기온(℃)"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "일시",
        "최고기온(℃)",
        "최저기온(℃)"
    ]
).copy()

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()

df["연도"] = df["일시"].dt.year
df["월"] = df["일시"].dt.month

# -----------------------------
# 5. 월 선택
# -----------------------------
month_names = {
    1: "1월", 2: "2월", 3: "3월",
    4: "4월", 5: "5월", 6: "6월",
    7: "7월", 8: "8월", 9: "9월",
    10: "10월", 11: "11월", 12: "12월"
}

selected_month = st.selectbox(
    "📅 분석할 월을 선택하세요.",
    options=list(range(1, 13)),
    format_func=lambda x: month_names[x]
)

# -----------------------------
# 6. 선택한 월의 연도별 데이터
# -----------------------------
monthly_df = df[
    df["월"] == selected_month
]

if monthly_df.empty:
    st.warning("선택한 월의 데이터가 없습니다.")
    st.stop()

# 같은 연도에 여러 자료가 있으면 평균 계산
yearly = (
    monthly_df
    .groupby("연도")
    .agg({
        "최고기온(℃)": "mean",
        "최저기온(℃)": "mean"
    })
    .reset_index()
    .sort_values("연도")
)

# -----------------------------
# 7. 한글 폰트 설정
# -----------------------------
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

# -----------------------------
# 8. 꺾은 선 그래프
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# 최고기온 - 형광 주황색
ax.plot(
    yearly["연도"],
    yearly["최고기온(℃)"],
    color="#FF5A00",
    marker="o",
    linewidth=2.5,
    markersize=5,
    label="최고기온"
)

# 최저기온 - 형광 파란색
ax.plot(
    yearly["연도"],
    yearly["최저기온(℃)"],
    color="#00BFFF",
    marker="o",
    linewidth=2.5,
    markersize=5,
    label="최저기온"
)

# 그래프 제목 및 축
ax.set_title(
    "월별 기온분석",
    fontsize=20,
    pad=20
)

ax.set_xlabel("연도", fontsize=13)
ax.set_ylabel("온도 (℃)", fontsize=13)

# 범례
ax.legend(
    loc="best",
    fontsize=12
)

# 격자
ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)

# 연도 눈금 조정
years = yearly["연도"].tolist()

if len(years) > 20:
    step = max(1, len(years) // 15)
    ax.set_xticks(years[::step])

plt.tight_layout()

# Streamlit 화면 출력
st.pyplot(fig, use_container_width=True)

plt.close(fig)

# -----------------------------
# 9. 데이터 표 출력
# -----------------------------
st.subheader(
    f"📊 {selected_month}월 연도별 기온 데이터"
)

yearly["최고기온(℃)"] = yearly[
    "최고기온(℃)"
].round(1)

yearly["최저기온(℃)"] = yearly[
    "최저기온(℃)"
].round(1)

st.dataframe(
    yearly,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "※ 동일 연도에 여러 자료가 있으면 평균값으로 계산합니다."
)
