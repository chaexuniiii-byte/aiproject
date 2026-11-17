
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. 기본 화면 설정
# --------------------------------------------------

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 꽃 품종 예측 AI")
st.write("꽃의 크기를 입력하면 인공지능이 아이리스 품종을 예측해 줍니다.")


# --------------------------------------------------
# 2. CSV 파일 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():

    # app.py가 있는 pages 폴더의 상위 폴더
    base_dir = Path(__file__).resolve().parent.parent

    # 상위 폴더에 있는 iris.csv
    csv_path = base_dir / "iris.csv"

    if not csv_path.exists():
        st.error(
            f"iris.csv 파일을 찾을 수 없습니다.\n\n"
            f"찾으려는 경로: {csv_path}"
        )
        st.stop()

    # 한글 CSV 인코딩 대응
    try:
        df = pd.read_csv(csv_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    return df


df = load_data()


# --------------------------------------------------
# 3. 데이터 확인
# --------------------------------------------------

feature_names = [
    "꽃받침 길이",
    "꽃받침 너비",
    "꽃잎 길이",
    "꽃잎 너비"
]

target_name = "품종"

# 필요한 열이 있는지 확인
required_columns = feature_names + [target_name]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(
        "CSV 파일에서 필요한 열을 찾지 못했습니다.\n\n"
        + ", ".join(missing_columns)
    )
    st.stop()

# 숫자 데이터 변환
for col in feature_names:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 4. 데이터 정보
# --------------------------------------------------

with st.expander("📊 아이리스 데이터 확인하기"):
    st.write("데이터 크기:", df.shape)
    st.dataframe(df.head(10), use_container_width=True)

    st.write("품종별 데이터 개수")
    st.dataframe(df[target_name].value_counts())


# --------------------------------------------------
# 5. 사용할 알고리즘
# --------------------------------------------------

def create_models():

    models = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "의사결정나무": DecisionTreeClassifier(
            random_state=42,
            max_depth=4
        ),

        "랜덤포레스트": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    return models


# --------------------------------------------------
# 6. 예측 및 시각화 함수
# --------------------------------------------------

def run_prediction(selected_features, input_values, tab_key):

    X = df[selected_features]
    y = df[target_name]

    models = create_models()

    results = []

    # 알고리즘별 학습 및 예측
    for model_name, model in models.items():

        model.fit(X, y)

        input_df = pd.DataFrame(
            [input_values],
            columns=selected_features
        )

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]
        max_probability = max(probabilities) * 100

        results.append({
            "알고리즘": model_name,
            "예측 품종": prediction,
            "예측 확률(%)": round(max_probability, 2)
        })

    result_df = pd.DataFrame(results)

    # --------------------------------------------------
    # 7. 예측 결과 출력
    # --------------------------------------------------

    st.subheader("🔍 알고리즘별 예측 결과")

    cols = st.columns(3)

    for i, result in enumerate(results):

        with cols[i]:
            st.metric(
                label=result["알고리즘"],
                value=result["예측 품종"],
                delta=f"예측 확률 {result['예측 확률(%)']}%"
            )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    # 예측 일치 여부
    predictions = [r["예측 품종"] for r in results]

    if len(set(predictions)) == 1:
        st.success("🎉 세 알고리즘의 예측 결과가 모두 일치합니다!")
    else:
        st.warning("⚠️ 알고리즘에 따라 예측 결과가 다릅니다.")

    # --------------------------------------------------
    # 8. 예측 확률 비교 그래프
    # --------------------------------------------------

    st.subheader("📈 알고리즘별 예측 확률 비교")

    probability_chart = result_df.set_index("알고리즘")[
        "예측 확률(%)"
    ]

    st.bar_chart(probability_chart)

    st.caption(
        "※ 예측 확률은 모델이 해당 품종이라고 판단한 정도이며, "
        "실제 예측 정확도와는 다릅니다."
    )

    # --------------------------------------------------
    # 9. 산점도
    # --------------------------------------------------

    st.subheader("🌸 꽃 특성 산점도")

    fig, ax = plt.subplots(figsize=(9, 5))

    varieties = df[target_name].unique()

    for variety in varieties:

        species_data = df[df[target_name] == variety]

        ax.scatter(
            species_data[selected_features[0]],
            species_data[selected_features[1]],
            label=variety,
            alpha=0.7
        )

    # 사용자 입력값: 노란색 별표
    ax.scatter(
        input_values[0],
        input_values[1],
        color="yellow",
        edgecolors="black",
        marker="*",
        s=350,
        label="입력한 꽃",
        zorder=5
    )

    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_title("아이리스 꽃 특성 분포")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption("⭐ 노란색 별표가 사용자가 입력한 꽃의 위치입니다.")

    # --------------------------------------------------
    # 10. 알고리즘 원리 설명
    # --------------------------------------------------

    st.subheader("🤖 인공지능 알고리즘 원리")

    with st.expander("1️⃣ KNN (K-최근접 이웃)"):
        st.write("""
        - 새로운 데이터와 기존 데이터 사이의 거리를 계산합니다.
        - 가장 가까운 5개의 데이터를 찾습니다.
        - 주변 이웃 중 가장 많은 품종을 새로운 꽃의 품종으로 예측합니다.
        - 데이터의 거리와 주변 분포가 예측에 영향을 줍니다.
        """)

    with st.expander("2️⃣ 의사결정나무 (Decision Tree)"):
        st.write("""
        - 데이터를 여러 조건에 따라 나누는 알고리즘입니다.
        - 꽃잎 길이, 꽃잎 너비 등의 기준으로 데이터를 분류합니다.
        - 질문에 답하듯 여러 단계를 거쳐 최종 품종을 결정합니다.
        - 나무 구조로 표현할 수 있어 원리를 이해하기 쉽습니다.
        """)

    with st.expander("3️⃣ 랜덤포레스트 (Random Forest)"):
        st.write("""
        - 여러 개의 의사결정나무를 만들어 예측합니다.
        - 각 나무의 예측 결과를 모아서 최종 품종을 결정합니다.
        - 여러 나무의 결과를 종합하기 때문에 하나의 나무보다
          안정적인 예측을 기대할 수 있습니다.
        """)


# --------------------------------------------------
# 11. 탭 구성
# --------------------------------------------------

tab1, tab2 = st.tabs([
    "🌷 탭 1. 꽃잎 특성으로 예측하기",
    "🌼 탭 2. 원하는 특성 2개 선택하기"
])


# --------------------------------------------------
# 탭 1: 꽃잎 길이와 꽃잎 너비
# --------------------------------------------------

with tab1:

    st.header("🌷 꽃잎 특성으로 품종 예측하기")

    selected_features = ["꽃잎 길이", "꽃잎 너비"]

    col1, col2 = st.columns(2)

    with col1:
        petal_length = st.slider(
            "꽃잎 길이 (cm)",
            float(df["꽃잎 길이"].min()),
            float(df["꽃잎 길이"].max()),
            float(df["꽃잎 길이"].mean()),
            0.1,
            key="tab1_length"
        )

    with col2:
        petal_width = st.slider(
            "꽃잎 너비 (cm)",
            float(df["꽃잎 너비"].min()),
            float(df["꽃잎 너비"].max()),
            float(df["꽃잎 너비"].mean()),
            0.1,
            key="tab1_width"
        )

    st.info(
        f"입력한 값: 꽃잎 길이 {petal_length}cm / "
        f"꽃잎 너비 {petal_width}cm"
    )

    if st.button("🔍 품종 예측하기", key="predict_tab1"):

        run_prediction(
            selected_features,
            [petal_length, petal_width],
            "tab1"
        )


# --------------------------------------------------
# 탭 2: 원하는 특성 2개 선택
# --------------------------------------------------

with tab2:

    st.header("🌼 원하는 꽃 특성 2개로 예측하기")

    col1, col2 = st.columns(2)

    with col1:
        feature1 = st.selectbox(
            "첫 번째 특성 선택",
            feature_names,
            index=0,
            key="tab2_feature1"
        )

    second_options = [
        feature for feature in feature_names
        if feature != feature1
    ]

    with col2:
        feature2 = st.selectbox(
            "두 번째 특성 선택",
            second_options,
            index=0,
            key="tab2_feature2"
        )

    st.write("### 📏 선택한 특성의 값 입력하기")

    col1, col2 = st.columns(2)

    with col1:
        value1 = st.slider(
            f"{feature1} (cm)",
            float(df[feature1].min()),
            float(df[feature1].max()),
            float(df[feature1].mean()),
            0.1,
            key="tab2_value1"
        )

    with col2:
        value2 = st.slider(
            f"{feature2} (cm)",
            float(df[feature2].min()),
            float(df[feature2].max()),
            float(df[feature2].mean()),
            0.1,
            key="tab2_value2"
        )

    st.info(
        f"선택한 특성: {feature1}, {feature2}\n\n"
        f"입력한 값: {value1} / {value2}"
    )

    if st.button("🔍 선택한 특성으로 예측하기", key="predict_tab2"):

        run_prediction(
            [feature1, feature2],
            [value1, value2],
            "tab2"
        )


# --------------------------------------------------
# 12. 마무리 안내
# --------------------------------------------------

st.divider()

st.caption(
    "🌸 아이리스 품종 예측 AI | "
    "KNN · Decision Tree · Random Forest"
)
