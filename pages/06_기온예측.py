
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# =====================================
# 1. 화면 설정
# =====================================
st.set_page_config(
    page_title="AI 기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 인공지능 회귀 알고리즘 기온 예측")
st.markdown("""
과거 기온 데이터를 이용하여 머신러닝 알고리즘으로
미래의 기온을 예측해 봅시다!
""")

st.divider()


# =====================================
# 2. CSV 파일 경로 설정
# =====================================
# 현재 파일: pages/06_기온예측.py
# CSV 파일: 프로젝트 최상위 폴더/temperature.csv

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "temperature.csv"


# =====================================
# 3. CSV 데이터 불러오기
# =====================================
@st.cache_data
def load_data():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다.\n{CSV_PATH}"
        )

    # 한글 인코딩 처리
    try:
        data = pd.read_csv(
            CSV_PATH,
            encoding="cp949"
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            CSV_PATH,
            encoding="utf-8-sig"
        )

    # 필요한 열 확인
    required_columns = [
        "일시",
        "평균기온(℃)"
    ]

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

    # 기온 숫자 변환
    data["평균기온(℃)"] = pd.to_numeric(
        data["평균기온(℃)"],
        errors="coerce"
    )

    # 결측치 제거
    data = data.dropna(
        subset=["날짜", "평균기온(℃)"]
    ).copy()

    # 연도와 월 분리
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


# =====================================
# 4. 사용자 입력
# =====================================
st.subheader("🔎 예측 조건 선택")

month_names = {
    1: "1월",
    2: "2월",
    3: "3월",
    4: "4월",
    5: "5월",
    6: "6월",
    7: "7월",
    8: "8월",
    9: "9월",
    10: "10월",
    11: "11월",
    12: "12월"
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


# =====================================
# 5. 회귀 알고리즘 선택
# =====================================
st.divider()

st.subheader("🤖 회귀 알고리즘 선택")

algorithm_names = {
    "선형회귀": "Linear Regression",
    "랜덤 포레스트 회귀": "Random Forest Regression",
    "그래디언트 부스팅 회귀": "Gradient Boosting Regression"
}

selected_algorithm = st.selectbox(
    "사용할 머신러닝 알고리즘을 선택하세요.",
    list(algorithm_names.keys())
)

algorithm_descriptions = {
    "선형회귀":
        "데이터의 전체적인 추세를 직선으로 학습합니다.",

    "랜덤 포레스트 회귀":
        "여러 의사결정나무의 예측값을 평균하여 결과를 만듭니다.",

    "그래디언트 부스팅 회귀":
        "여러 작은 의사결정나무를 순차적으로 학습하여 오차를 보완합니다."
}

st.info(
    f"선택한 알고리즘: **{selected_algorithm}**\n\n"
    f"{algorithm_descriptions[selected_algorithm]}"
)


# =====================================
# 6. 선택한 월의 데이터 추출
# =====================================
month_df = df[
    df["월"] == selected_month
].copy()

month_df = month_df.sort_values("연도")

if len(month_df) < 3:
    st.error("회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()


# =====================================
# 7. 세 가지 회귀 모델 생성
# =====================================
X = month_df[["연도"]]
y = month_df["평균기온(℃)"]

models = {

    "선형회귀": LinearRegression(),

    "랜덤 포레스트 회귀": RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    ),

    "그래디언트 부스팅 회귀": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=42
    )
}


# =====================================
# 8. 모든 알고리즘 학습 및 예측
# =====================================
results = {}
models_info = {}

for name, model in models.items():

    # 모델 학습
    model.fit(X, y)

    # 과거 데이터 예측
    train_predictions = model.predict(X)

    # 미래 데이터 예측
    future_prediction = model.predict(
        pd.DataFrame({
            "연도": [predict_year]
        })
    )[0]

    # 결정계수
    r2 = model.score(X, y)

    results[name] = {
        "예측기온": future_prediction,
        "결정계수": r2
    }

    models_info[name] = model


# =====================================
# 9. 선택한 알고리즘 결과
# =====================================
selected_model = models_info[selected_algorithm]

prediction = results[selected_algorithm]["예측기온"]

r2 = results[selected_algorithm]["결정계수"]

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
        f"{prediction:.2f} ℃"
    )

st.success(
    f"🌡️ {selected_algorithm} 분석 결과\n\n"
    f"{predict_year}년 {month_names[selected_month]}의 "
    f"예상 평균기온은 **{prediction:.2f}℃**입니다."
)


# =====================================
# 10. 선택한 알고리즘 그래프
# =====================================
st.divider()

st.subheader(
    f"📈 {month_names[selected_month]} "
    f"{selected_algorithm} 예측 그래프"
)

# 그래프 연도 설정
start_year = int(month_df["연도"].min())

end_year = max(
    int(month_df["연도"].max()),
    predict_year
)

line_years = np.arange(
    start_year,
    end_year + 1
)

line_predictions = selected_model.predict(
    pd.DataFrame({
        "연도": line_years
    })
)

# 과거 데이터 예측
month_df["회귀예측"] = selected_model.predict(X)

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 6))

# 실제 기온
ax.scatter(
    month_df["연도"],
    month_df["평균기온(℃)"],
    color="royalblue",
    s=45,
    alpha=0.7,
    label="Actual Temperature",
    zorder=3
)

# 실제값과 예측값의 차이
ax.vlines(
    month_df["연도"],
    month_df["평균기온(℃)"],
    month_df["회귀예측"],
    color="gray",
    alpha=0.4,
    linewidth=1,
    label="Prediction Error"
)

# 회귀 모델의 예측선
ax.plot(
    line_years,
    line_predictions,
    color="red",
    linewidth=2.5,
    label=selected_algorithm
)

# 미래 예측 지점
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
    f"{prediction:.2f}℃",
    xy=(predict_year, prediction),
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=12,
    fontweight="bold",
    color="green"
)

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


# =====================================
# 11. 세 가지 알고리즘 결과 비교
# =====================================
st.divider()

st.subheader("🏆 세 가지 알고리즘 예측 결과 비교")

st.write(
    "동일한 월과 예측연도를 기준으로 "
    "세 가지 알고리즘의 결과를 비교합니다."
)

comparison_df = pd.DataFrame([
    {
        "알고리즘": name,
        "예측 평균기온(℃)": round(
            info["예측기온"], 2
        ),
        "결정계수(R²)": round(
            info["결정계수"], 4
        )
    }
    for name, info in results.items()
])

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)


# =====================================
# 12. 알고리즘별 예측값 비교 그래프
# =====================================
st.subheader("📊 알고리즘별 예측 기온 비교")

fig2, ax2 = plt.subplots(figsize=(10, 5))

algorithm_list = list(results.keys())

prediction_list = [
    results[name]["예측기온"]
    for name in algorithm_list
]

bars = ax2.bar(
    algorithm_list,
    prediction_list
)

# 막대 위에 기온 표시
for bar, value in zip(bars, prediction_list):

    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}℃",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax2.set_title(
    f"{predict_year}년 {month_names[selected_month]} "
    "알고리즘별 예측 결과"
)

ax2.set_ylabel("Predicted Temperature (℃)")

ax2.grid(
    axis="y",
    alpha=0.25
)

fig2.tight_layout()

st.pyplot(fig2)

plt.close(fig2)


# =====================================
# 13. 알고리즘별 차이 확인
# =====================================
st.subheader("🔍 알고리즘별 예측 차이")

max_prediction = max(prediction_list)
min_prediction = min(prediction_list)

difference = max_prediction - min_prediction

st.metric(
    "세 알고리즘의 최대 예측 차이",
    f"{difference:.2f} ℃"
)

st.write(
    f"세 가지 알고리즘 중 가장 높은 예측값과 "
    f"가장 낮은 예측값의 차이는 **{difference:.2f}℃**입니다."
)

st.caption(
    "알고리즘마다 학습 방법과 예측 방식이 다르기 때문에 "
    "동일한 데이터에서도 서로 다른 결과가 나올 수 있습니다."
)


# =====================================
# 14. 선택한 알고리즘의 분석 정보
# =====================================
st.divider()

st.subheader("🧠 선택한 알고리즘 분석 정보")

st.metric(
    "결정계수 (R²)",
    f"{r2:.4f}"
)

st.markdown("""
**결정계수(R²)란?**

실제 기온 데이터의 변동을 모델이 얼마나 설명하는지를 나타내는 지표입니다.

- 1에 가까울수록 학습 데이터에 대한 설명력이 높습니다.
- 0에 가까우면 평균값을 사용하는 것과 비슷한 설명력을 가집니다.
- 음수라면 학습 데이터에서 평균값을 사용하는 것보다 설명력이 낮습니다.

주의: 여기 표시된 결정계수는 학습에 사용한 데이터에 대한 값이며,
미래 예측의 정확도를 보장하지 않습니다.
""")


# =====================================
# 15. 알고리즘 작동 원리 설명
# =====================================
st.divider()

st.subheader("📚 세 가지 회귀 알고리즘 알아보기")

# 선형회귀
with st.expander("① 선형회귀 (Linear Regression)"):

    st.markdown("""
    **데이터를 가장 잘 설명하는 직선을 찾는 알고리즘입니다.**

    연도와 기온 사이의 관계를 직선으로 표현합니다.

    예측 수식:
    """)

    st.latex(
        r"\hat{y} = b_0 + b_1x"
    )

    st.markdown("""
    - x : 연도
    - b₀ : 절편
    - b₁ : 기울기
    - ŷ : 예측 기온

    **특징**

    - 전체적인 상승 및 하락 추세를 파악하기 좋습니다.
    - 계산이 빠르고 결과를 이해하기 쉽습니다.
    - 복잡한 곡선 형태의 관계를 표현하기 어렵습니다.
    """)


# 랜덤 포레스트
with st.expander("② 랜덤 포레스트 회귀 (Random Forest)"):

    st.markdown("""
    **여러 개의 의사결정나무를 만들어 결과를 평균내는 알고리즘입니다.**

    하나의 나무만 사용하는 것이 아니라 여러 나무가 각각 예측한 결과를 모아 최종 예측값을 만듭니다.

    예측 개념:
    """)

    st.latex(
        r"\hat{y} = \frac{1}{N}\sum_{i=1}^{N}T_i(x)"
    )

    st.markdown("""
    - N : 의사결정나무의 개수
    - Tᵢ(x) : 각 나무의 예측 결과
    - ŷ : 최종 예측값

    **특징**

    - 여러 나무의 결과를 평균하여 예측합니다.
    - 비선형적인 관계를 학습할 수 있습니다.
    - 데이터의 복잡한 패턴을 찾는 데 유용합니다.
    - 일반적으로 학습 데이터의 범위를 벗어난 미래 외삽에는 적합하지 않습니다.
    """)


# 그래디언트 부스팅
with st.expander("③ 그래디언트 부스팅 회귀 (Gradient Boosting)"):

    st.markdown("""
    **여러 개의 작은 의사결정나무를 순서대로 학습하는 알고리즘입니다.**

    첫 번째 모델이 예측한 결과의 오차를 다음 모델이 보완합니다.

    이 과정을 반복하여 예측 성능을 개선합니다.

    **작동 과정**

    1. 첫 번째 모델이 기온을 예측합니다.
    2. 실제 기온과 예측 기온의 오차를 계산합니다.
    3. 다음 모델이 이전 모델의 오차를 보완합니다.
    4. 여러 모델의 결과를 합쳐 최종 예측값을 만듭니다.

    **특징**

    - 이전 모델의 오차를 순차적으로 보완합니다.
    - 복잡한 데이터 패턴을 학습할 수 있습니다.
    - 학습률과 나무의 깊이 등에 따라 결과가 달라집니다.
    - 과적합에 주의해야 합니다.
    """)


# =====================================
# 16. 데이터 확인
# =====================================
st.divider()

with st.expander("📋 분석에 사용한 원본 데이터 확인"):

    display_df = month_df[
        ["연도", "평균기온(℃)", "회귀예측"]
    ].copy()

    display_df.columns = [
        "연도",
        "실제 평균기온(℃)",
        "선택한 알고리즘 예측기온(℃)"
    ]

    display_df["선택한 알고리즘 예측기온(℃)"] = (
        display_df[
            "선택한 알고리즘 예측기온(℃)"
        ].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# =====================================
# 17. 마무리
# =====================================
st.divider()

st.caption(
    "🌍 AI 기온 예측 | Linear Regression · "
    "Random Forest · Gradient Boosting"
)
