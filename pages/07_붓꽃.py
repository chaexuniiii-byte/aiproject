
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 붓꽃 품종 예측")
st.write(
    "머신러닝 알고리즘과 꽃의 측정 속성을 선택하여 "
    "붓꽃의 품종을 예측해 보세요!"
)

st.divider()


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "iris.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "아이리스.csv 파일이 app.py와 같은 폴더에 있어야 합니다."
        )

    try:
        data = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="utf-8-sig")

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# ==========================================
# 3. 데이터 정리
# ==========================================

features = [
    "꽃받침 길이",
    "꽃받침 너비",
    "꽃잎 길이",
    "꽃잎 너비"
]

target = "품종"

required_columns = features + [target]

if not all(col in df.columns for col in required_columns):
    st.error("CSV 파일의 열 이름을 확인해주세요.")
    st.stop()

df = df[required_columns].copy()

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)

if len(df) < 5:
    st.error("분석할 데이터가 부족합니다.")
    st.stop()


# ==========================================
# 4. 품종 이름
# ==========================================

korean_names = {
    "Iris-setosa": "Setosa (세토사)",
    "Iris-versicolor": "Versicolor (버시컬러)",
    "Iris-virginica": "Virginica (버지니카)"
}


def species_name(name):
    return korean_names.get(name, name)


# ==========================================
# 5. 알고리즘 설명
# ==========================================

algorithm_info = {

    "KNN (K-최근접 이웃)": {
        "principle": "가장 가까운 5개의 데이터를 찾아 품종을 예측합니다.",
        "detail": """
        1. 사용자가 꽃잎 또는 꽃받침 크기를 입력합니다.

        2. 기존 붓꽃 데이터와 입력값 사이의 거리를 계산합니다.

        3. 가장 가까운 5개의 데이터를 찾습니다.

        4. 가까운 데이터의 품종을 참고하여 최종 품종을 결정합니다.
        """,
        "example": "주변 이웃의 정보를 참고하는 방식"
    },

    "Decision Tree (의사결정나무)": {
        "principle": "측정값을 여러 조건으로 나누어 품종을 예측합니다.",
        "detail": """
        1. 꽃의 측정값을 확인합니다.

        2. 꽃의 크기에 따라 조건을 나눕니다.

        3. 조건을 따라 여러 단계로 이동합니다.

        4. 마지막 단계에서 붓꽃 품종을 결정합니다.
        """,
        "example": "질문에 답하면서 정답을 찾아가는 방식"
    },

    "Random Forest (랜덤 포레스트)": {
        "principle": "여러 개의 의사결정나무가 예측한 결과를 종합합니다.",
        "detail": """
        1. 여러 개의 의사결정나무를 만듭니다.

        2. 각 나무가 입력된 꽃의 품종을 예측합니다.

        3. 각각의 예측 결과를 모읍니다.

        4. 여러 나무의 판단을 종합하여 최종 품종을 결정합니다.
        """,
        "example": "여러 전문가의 의견을 모아 결정하는 방식"
    }
}


# ==========================================
# 6. 알고리즘 모델 생성
# ==========================================

algorithm_names = list(algorithm_info.keys())


def create_model(algorithm):

    if algorithm == algorithm_names[0]:

        classifier = KNeighborsClassifier(
            n_neighbors=min(5, len(df)),
            weights="distance"
        )

    elif algorithm == algorithm_names[1]:

        classifier = DecisionTreeClassifier(
            random_state=42
        )

    else:

        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])


# ==========================================
# 7. 데이터 정보
# ==========================================

st.subheader("📊 아이리스 데이터 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df)}개")
col2.metric("붓꽃 품종", f"{df[target].nunique()}종")
col3.metric("분석 속성", "4개")

with st.expander("📋 데이터 미리보기"):
    st.dataframe(df.head(10), use_container_width=True)


# ==========================================
# 8. 알고리즘 선택
# ==========================================

st.divider()

st.subheader("🤖 사용할 알고리즘 선택")

algorithm = st.selectbox(
    "예측에 사용할 알고리즘을 선택하세요.",
    algorithm_names
)

st.info(algorithm_info[algorithm]["principle"])


# ==========================================
# 9. 예측 및 비교 함수
# ==========================================

def run_prediction(selected_features, input_values):

    X = df[selected_features]
    y = df[target]

    input_df = pd.DataFrame(
        [input_values],
        columns=selected_features
    )

    # --------------------------------------
    # 선택한 알고리즘으로 예측
    # --------------------------------------

    model = create_model(algorithm)
    model.fit(X, y)

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    classes = model.named_steps["classifier"].classes_

    st.divider()
    st.subheader("🎉 선택한 알고리즘의 예측 결과")

    st.success(
        f"예측 품종: **{species_name(prediction)}**"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            selected_features[0],
            f"{input_values[0]:.1f} cm"
        )

    with col2:
        st.metric(
            selected_features[1],
            f"{input_values[1]:.1f} cm"
        )

    st.subheader("📈 품종별 예측 확률")

    for name, probability in zip(classes, probabilities):

        st.write(
            f"**{species_name(name)}: "
            f"{probability * 100:.1f}%**"
        )

        st.progress(float(probability))


    # ======================================
    # 10. 산점도
    # ======================================

    st.divider()
    st.subheader("📍 입력값 산점도")

    st.write(
        f"가로축: {selected_features[0]} / "
        f"세로축: {selected_features[1]}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for species in sorted(df[target].unique()):

        species_data = df[df[target] == species]

        ax.scatter(
            species_data[selected_features[0]],
            species_data[selected_features[1]],
            label=species,
            s=55,
            alpha=0.7
        )

    # 사용자 입력값 노란색 별
    ax.scatter(
        input_values[0],
        input_values[1],
        color="yellow",
        edgecolors="black",
        marker="*",
        s=500,
        linewidths=1.5,
        label="Your Input",
        zorder=10
    )

    ax.set_xlabel(
        selected_features[0] + " (cm)"
    )

    ax.set_ylabel(
        selected_features[1] + " (cm)"
    )

    ax.set_title("Iris Classification Scatter Plot")

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


    # ======================================
    # 11. 그래프 하단 알고리즘 작동 원리
    # ======================================

    st.divider()

    st.subheader("🧠 알고리즘 작동 원리")

    st.write(
        f"현재 선택한 알고리즘: **{algorithm}**"
    )

    info = algorithm_info[algorithm]

    st.success(info["principle"])

    st.markdown(info["detail"])

    st.caption(
        "쉽게 이해하기: " + info["example"]
    )


    # ======================================
    # 12. 세 가지 알고리즘 결과 비교
    # ======================================

    st.divider()

    st.subheader("🏆 세 가지 알고리즘 예측 결과 비교")

    st.write(
        "동일한 입력값과 동일한 속성 2개를 사용하여 "
        "세 가지 알고리즘의 예측 결과를 비교합니다."
    )

    comparison_results = []

    for name in algorithm_names:

        compare_model = create_model(name)

        compare_model.fit(X, y)

        compare_prediction = compare_model.predict(
            input_df
        )[0]

        compare_probabilities = compare_model.predict_proba(
            input_df
        )[0]

        max_probability = max(compare_probabilities) * 100

        comparison_results.append({

            "알고리즘": name,

            "예측 품종": species_name(
                compare_prediction
            ),

            "예측 확률(%)": round(
                max_probability, 1
            ),

            "예측 결과": compare_prediction

        })


    # 비교 결과 표
    comparison_df = pd.DataFrame(
        comparison_results
    )

    # 선택한 알고리즘 표시
    display_df = comparison_df[
        ["알고리즘", "예측 품종", "예측 확률(%)"]
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ======================================
    # 13. 예측 확률 그래프
    # ======================================

    st.subheader("📊 알고리즘별 예측 확률 비교")

    chart_df = comparison_df[
        ["알고리즘", "예측 확률(%)"]
    ].set_index("알고리즘")

    st.bar_chart(
        chart_df,
        y="예측 확률(%)"
    )


    # ======================================
    # 14. 예측 결과 일치 여부
    # ======================================

    st.subheader("🔍 알고리즘 예측 결과 분석")

    predicted_species = comparison_df[
        "예측 결과"
    ].tolist()

    unique_predictions = set(predicted_species)

    if len(unique_predictions) == 1:

        st.success(
            "🎉 세 가지 알고리즘이 모두 "
            "동일한 품종을 예측했습니다!"
        )

    else:

        st.warning(
            "알고리즘에 따라 예측 품종이 다릅니다. "
            "각 알고리즘의 판단 기준을 비교해 보세요."
        )

    # 알고리즘별 예측 품종
    for _, row in comparison_df.iterrows():

        st.write(
            f"**{row['알고리즘']}** → "
            f"{species_name(row['예측 결과'])}"
        )

    st.caption(
        "주의: 예측 확률은 각 모델의 추정값이며 "
        "실제 정답률이나 정확도를 의미하지 않습니다."
    )


# ==========================================
# 15. 탭 구성
# ==========================================

tab1, tab2 = st.tabs([
    "🌷 꽃잎 속성으로 예측하기",
    "🌿 원하는 속성 2개 선택하기"
])


# ==========================================
# TAB 1
# ==========================================

with tab1:

    st.header("🌷 꽃잎 속성으로 품종 예측")

    st.write(
        "꽃잎 길이와 꽃잎 너비를 입력하세요."
    )

    feature1 = "꽃잎 길이"
    feature2 = "꽃잎 너비"

    col1, col2 = st.columns(2)

    with col1:

        min1 = float(df[feature1].min())
        max1 = float(df[feature1].max())

        value1 = st.slider(
            "꽃잎 길이 (cm)",
            min1,
            max1,
            float(df[feature1].median()),
            step=0.1,
            key="petal_length"
        )

        st.caption(
            f"입력 범위: {min1} ~ {max1} cm"
        )

    with col2:

        min2 = float(df[feature2].min())
        max2 = float(df[feature2].max())

        value2 = st.slider(
            "꽃잎 너비 (cm)",
            min2,
            max2,
            float(df[feature2].median()),
            step=0.1,
            key="petal_width"
        )

        st.caption(
            f"입력 범위: {min2} ~ {max2} cm"
        )

    if st.button(
        "🔍 꽃잎 속성으로 예측하기",
        key="predict1",
        use_container_width=True
    ):

        run_prediction(
            [feature1, feature2],
            [value1, value2]
        )


# ==========================================
# TAB 2
# ==========================================

with tab2:

    st.header("🌿 원하는 속성 2개로 품종 예측")

    st.write(
        "네 가지 속성 중 원하는 두 가지를 선택하세요."
    )

    feature1 = st.selectbox(
        "① 첫 번째 속성",
        features,
        index=0,
        key="feature_select1"
    )

    remaining_features = [
        f for f in features if f != feature1
    ]

    feature2 = st.selectbox(
        "② 두 번째 속성",
        remaining_features,
        index=0,
        key="feature_select2"
    )

    st.success(
        f"선택한 속성: {feature1} + {feature2}"
    )

    col1, col2 = st.columns(2)

    with col1:

        min1 = float(df[feature1].min())
        max1 = float(df[feature1].max())

        input1 = st.slider(
            f"{feature1} (cm)",
            min1,
            max1,
            float(df[feature1].median()),
            step=0.1,
            key="custom_value1"
        )

        st.caption(
            f"입력 범위: {min1} ~ {max1} cm"
        )

    with col2:

        min2 = float(df[feature2].min())
        max2 = float(df[feature2].max())

        input2 = st.slider(
            f"{feature2} (cm)",
            min2,
            max2,
            float(df[feature2].median()),
            step=0.1,
            key="custom_value2"
        )

        st.caption(
            f"입력 범위: {min2} ~ {max2} cm"
        )

    if st.button(
        "🔍 선택한 속성 2개로 예측하기",
        key="predict2",
        use_container_width=True
    ):

        run_prediction(
            [feature1, feature2],
            [input1, input2]
        )


# ==========================================
# 16. 하단 안내
# ==========================================

st.divider()

st.subheader("💡 머신러닝 학습 안내")

st.markdown("""
- **KNN:** 주변의 가까운 데이터를 참고합니다.
- **Decision Tree:** 조건을 나누어 품종을 구분합니다.
- **Random Forest:** 여러 의사결정나무의 결과를 종합합니다.

알고리즘마다 학습 방식이 다르므로 예측 결과도 달라질 수 있습니다.

입력 속성을 바꾸어 가면서 세 알고리즘의 결과를 비교해 보세요!
""")

st.caption(
    "아이리스 데이터 분석 및 머신러닝 실습 프로그램"
)
