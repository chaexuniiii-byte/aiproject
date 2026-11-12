
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ---------------------------------------
# 1. 화면 설정
# ---------------------------------------
st.set_page_config(
    page_title="AI 기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 선형회귀를 이용한 기온 예측 AI")
st.write("과거 기온 데이터를 학습하여 미래의 기온을 예측해 봅시다!")

st.divider()


# ---------------------------------------
# 2. CSV 파일 경로 설정
# ---------------------------------------
# 현재 파일은 pages 폴더 안에 있음
# CSV 파일은 pages의 상위 폴더에 있음

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "temperature.csv"


# ---------------------------------------
# 3. CSV 데이터 불러오기
# ---------------------------------------
@st.cache_data
def load_data():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다.\n경로: {CSV_PATH}"
        )

    # 한글 인코딩 대응
    try:
        data = pd.read_csv(CSV_PATH, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # 필요한 열 확인
    required_columns = ["일시", "평균기온(℃)"]

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(
                f"CSV 파일에 '{col}' 열이 없습니다."
            )

    # 날짜 변환: Oct-07 → 2007년 10월
    data["날짜"] = pd.to_datetime(
        data["일시"].astype(str).str.strip(),
        format="%b-%y",
        errors="coerce"
    )

    data["평균기온(℃)"] = pd.to_numeric(
        data["평균기온(℃)"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["날짜", "평균기온(℃)"]
    ).copy()

    data["연도"] = data["날짜"].dt.year
    data["월"] = data["날짜"].dt.month

    data["연도"] = data["연도"].astype(int)
    data["월"] = data["월"].astype(int)

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# ---------------------------------------
# 4. 사용자 입력
# ---------------------------------------
st.subheader("🔎 예측 조건 선택")

month_names = {
    1: "1월", 2: "2월", 3: "3월",
    4: "4월", 5: "5월", 6: "6월",
    7: "7월", 8: "8월", 9: "9월",
    10: "10월", 11: "11월", 12: "12월"
}

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox(
        "예측할 월을 선택하세요.",
        options=list(month_names.keys()),
        format_func=lambda x: month_names[x]
    )

with col2:
    min_year = int(df["연도"].min())
    max_year = int(df["연도"].max())

    predict_year = st.number_input(
        "예측할 연도를 입력하세요.",
        min_value=min_year,
        max_value=2100,
        value=max_year + 1,
        step=1
    )

predict_year = int(predict_year)


# ---------------------------------------
# 5. 선택한 월의 데이터 추출
# ---------------------------------------
month_df = df[
    df["월"] == selected_month
].copy()

month_df = month_df.sort_values("연도")

if len(month_df) < 2:
    st.error("예측에 필요한 데이터가 부족합니다.")
    st.stop()


# ---------------------------------------
# 6. 선형회귀 모델 학습
# ---------------------------------------
X = month_df[["연도"]]
y = month_df["평균기온(℃)"]

model = LinearRegression()

# 과거 데이터 학습
model.fit(X, y)

# 과거 데이터의 회귀 예측값
month_df["회귀예측"] = model.predict(X)

# 미래 기온 예측
prediction = model.predict(
    pd.DataFrame({"연도": [predict_year]})
)[0]

# 회귀식 정보
slope = model.coef_[0]
intercept = model.intercept_

# 결정계수
r2 = model.score(X, y)


# ---------------------------------------
# 7. 예측 결과 출력
# ---------------------------------------
st.divider()

st.subheader("🎯 AI 기온 예측 결과")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "선택한 월",
        month_names[selected_month]
    )

with c2:
    st.metric(
        "예측 연도",
        f"{predict_year}년"
    )

with c3:
    st.metric(
        "예측 평균기온",
        f"{prediction:.1f} ℃"
    )

st.success(
    f"🌡️ {predict_year}년 {month_names[selected_month]} "
    f"예상 평균기온은 {prediction:.1f}℃입니다."
)


# ---------------------------------------
# 8. 기온 그래프
# ---------------------------------------
st.divider()

st.subheader(
    f"📈 {month_names[selected_month]} 연도별 기온 및 예측 그래프"
)

# 회귀 직선에 사용할 연도
start_year = int(month_df["연도"].min())

end_year = max(
    int(month_df["연도"].max()),
    predict_year
)

line_years = np.arange(
    start_year,
    end_year + 1
)

line_predictions = model.predict(
    pd.DataFrame({"연도": line_years})
)

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 6))

# ① 실제 기온
ax.scatter(
    month_df["연도"],
    month_df["평균기온(℃)"],
    color="royalblue",
    s=45,
    alpha=0.7,
    label="Actual Temperature",
    zorder=3
)

# ② 실제값과 회귀선 사이의 오차
ax.vlines(
    month_df["연도"],
    month_df["평균기온(℃)"],
    month_df["회귀예측"],
    color="gray",
    alpha=0.4,
    linewidth=1.2,
    label="Prediction Error"
)

# ③ 선형회귀 직선
ax.plot(
    line_years,
    line_predictions,
    color="red",
    linewidth=2.5,
    label="Linear Regression Line"
)

# ④ 미래 예측 지점
ax.scatter(
    predict_year,
    prediction,
    color="limegreen",
    edgecolors="black",
    s=200,
    marker="*",
    label="Future Prediction",
    zorder=5
)

# 예측값 표시
ax.annotate(
    f"{prediction:.1f}℃",
    xy=(predict_year, prediction),
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=12,
    fontweight="bold",
    color="green"
)

# 예측 연도 수직선
ax.axvline(
    predict_year,
    color="green",
    linestyle="--",
    alpha=0.5
)

ax.set_title(
    f"{month_names[selected_month]} Temperature Prediction",
    fontsize=16
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Average Temperature (℃)", fontsize=12)

ax.grid(alpha=0.25)
ax.legend(loc="best")

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------
# 9. 선형회귀 분석 결과
# ---------------------------------------
st.divider()

st.subheader("🧠 AI 선형회귀 분석 정보")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "기울기 (Slope)",
        f"{slope:.4f}"
    )

with c2:
    st.metric(
        "결정계수 (R²)",
        f"{r2:.4f}"
    )

st.markdown("### 📐 선형회귀 예측 공식")

st.latex(
    r"\hat{y} = {:.4f}x + {:.4f}".format(
        slope, intercept
    )
)

st.write(f"""
- x : 연도
- ŷ : 예측 평균기온
- 기울기 : 연도가 1년 증가할 때 예측 기온의 변화량
- 결정계수(R²) : 회귀 직선이 과거 기온의 변동을 설명하는 정도
""")

if slope > 0:
    st.info(
        f"📈 분석 결과, 연평균적으로 해당 월의 예측 기온이 "
        f"약 {slope:.3f}℃씩 상승하는 추세입니다."
    )
elif slope < 0:
    st.info(
        f"📉 분석 결과, 연평균적으로 해당 월의 예측 기온이 "
        f"약 {abs(slope):.3f}℃씩 하락하는 추세입니다."
    )
else:
    st.info("분석 결과, 뚜렷한 상승 또는 하락 추세가 없습니다.")


# ---------------------------------------
# 10. 선형회귀 알고리즘 설명
# ---------------------------------------
st.divider()

st.subheader("📚 선형회귀(Linear Regression)란?")

st.markdown("""
선형회귀는 과거 데이터를 분석하여 데이터의 관계를
가장 잘 설명하는 직선을 찾는 머신러닝 알고리즘입니다.

이번 앱에서는 **연도와 평균기온의 관계**를 학습합니다.

### ① 데이터 학습

CSV 파일에서 사용자가 선택한 월의 모든 연도별 평균기온을 가져옵니다.

예를 들어 1월을 선택하면 모든 연도의 1월 평균기온을 학습합니다.

### ② 가장 적합한 직선 찾기

AI는 실제 기온과 직선이 예측한 기온의 차이를 계산합니다.

이 차이를 오차(Residual)라고 합니다.

오차의 제곱을 모두 더한 값이 최소가 되도록 직선을 결정합니다.

### ③ 미래 기온 예측

학습한 직선에 사용자가 입력한 연도를 대입합니다.

그 결과 해당 연도의 예상 평균기온을 계산합니다.

### ④ 그래프 읽는 방법

- 🔵 파란 점 : 실제 관측된 평균기온
- 🔴 빨간 선 : AI가 학습한 선형회귀 직선
- 🟢 초록 별 : 미래 예측 기온
- 회색 세로선 : 실제 기온과 회귀선의 예측값 차이

### ⑤ 선형회귀의 핵심 수식

""")

st.latex(
    r"\hat{y} = b_0 + b_1x"
)

st.markdown("""
- x : 입력 데이터(연도)
- b₀ : 절편
- b₁ : 기울기
- ŷ : 예측 결과(기온)

### ⚠️ 주의사항

선형회귀는 과거의 추세를 바탕으로 미래를 예측합니다.

실제 기온은 온실가스, 기후변화, 자연현상 등
다양한 요인의 영향을 받습니다.

따라서 예측 결과는 실제 기온과 다를 수 있으며,
특히 먼 미래를 예측할수록 오차가 커질 수 있습니다.
""")

st.divider()


# ---------------------------------------
# 11. 원본 데이터 확인
# ---------------------------------------
with st.expander("📋 분석에 사용한 원본 데이터 확인"):

    display_df = month_df[
        ["연도", "평균기온(℃)", "회귀예측"]
    ].copy()

    display_df.columns = [
        "연도",
        "실제 평균기온(℃)",
        "회귀선 예측기온(℃)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------
# 12. 마무리
# ---------------------------------------
st.divider()

st.caption(
    "🌍 기온 예측 AI | 선형회귀 머신러닝 데이터 분석 프로젝트"
)
