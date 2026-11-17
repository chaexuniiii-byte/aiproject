
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스(Iris) 꽃 품종 예측")
st.write(
    "꽃의 길이와 너비를 입력하면 머신러닝 알고리즘이 "
    "아이리스 꽃의 품종을 예측합니다."
)

st.divider()


# --------------------------------------------------
# 2. CSV 파일 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():

    # pages 폴더의 상위 폴더
    base_dir = Path(__file__).resolve().parent.parent

    # CSV 파일 경로
    csv_path = base_dir / "Iris.csv"

    if not csv_path.is_file():
        st.error("❌ Iris.csv 파일을 찾을 수 없습니다.")
        st.write("찾는 경로:", str(csv_path))
        st.write("현재 상위 폴더의 파일 목록:")

        if base_dir.exists():
            st.write([p.name for p in base_dir.iterdir()])

        st.stop()

    # CSV 파일 읽기
    try:
        df = pd.read_csv(csv_path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 열 이름 앞뒤 공백 제거
    df.columns = df.columns.str.strip()

    # 실제 CSV 열 이름을 영어로 변경
    df = df.rename(columns={
        "꽃받침 길이": "sepal_length",
        "꽃받침 너비": "sepal_width",
        "꽃잎 길이": "petal_length",
        "꽃잎 너비": "petal_width",
        "품종": "species"
    })

    # 필요한 열 확인
    required_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error("CSV 파일에서 필요한 열을 찾지 못했습니다.")
        st.write("찾지 못한 열:", missing_columns)
        st.write("실제 열 이름:", list(df.columns))
        st.stop()

    # 숫자형 데이터 변환
    for col in required_columns[:4]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)

    return df


df = load_data()


# --------------------------------------------------
# 3. 데이터 정보
# --------------------------------------------------

feature_columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

feature_names = {
    "sepal_length": "꽃받침 길이",
    "sepal_width": "꽃받침 너비",
    "petal_length": "꽃잎 길이",
    "petal_width": "꽃잎 너비"
}

species_names = {
    "Iris-setosa": "세토사",
    "Iris-versicolor": "버시컬러",
    "Iris-virginica": "버지니카"
}

X_all = df[feature_columns]
y = df["species"]

species_list = sorted(df["species"].unique())


# --------------------------------------------------
# 4. 머신러닝 알고리즘 생성
# --------------------------------------------------

def create_models():

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),

        "의사결정나무": DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

        "랜덤포레스트": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    return models


# --------------------------------------------------
# 5. 예측 및 그래프 함수
# --------------------------------------------------

def run_prediction(feature_x, feature_y, value_x, value_y):

    st.subheader("🤖 머신러닝 예측 결과")

    # 선택한 두 특성으로 학습
    X = df[[feature_x, feature_y]]

    # 입력 데이터
    input_data = pd.DataFrame(
        [[value_x, value_y]],
        columns=[feature_x, feature_y]
    )

    models = create_models()

    results = []
    prediction_data = []

    # 각 알고리즘 실행
    for model_name, model in models.items():

        model.fit(X, y)

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = max(probabilities) * 100

        results.append({
            "알고리즘": model_name,
            "예측 품종": species_names.get(
                prediction, prediction
            ),
            "예측 확률": round(confidence, 2)
        })

        prediction_data.append({
            "알고리즘": model_name,
            "예측 확률(%)": confidence,
            "예측 품종": species_names.get(
                prediction, prediction
            )
        })

    # 결과 표
    result_df = pd.DataFrame(results)

    col1, col2, col3 = st.columns(3)

    for col, row in zip(
        [col1, col2, col3],
        results
    ):
        with col:
            st.metric(
                label=row["알고리즘"],
                value=row["예측 품종"],
                delta=f"예측 확률 {row['예측 확률']}%"
            )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    # 모든 알고리즘의 예측 결과가 같은지 확인
    predictions = [r["예측 품종"] for r in results]

    if len(set(predictions)) == 1:
        st.success("✅ 세 알고리즘의 예측 결과가 모두 같습니다!")
    else:
        st.warning("⚠️ 알고리즘에 따라 예측 결과가 다릅니다.")

    # 예측 확률 비교 그래프
    st.subheader("📊 알고리즘별 예측 확률 비교")

    confidence_df = pd.DataFrame(prediction_data)

    confidence_chart = confidence_df.set_index("알고리즘")[
        ["예측 확률(%)"]
    ]

    st.bar_chart(confidence_chart)

    # --------------------------------------------------
    # 산점도
    # --------------------------------------------------

    st.subheader("🌸 꽃의 특성 산점도")

    fig, ax = plt.subplots(figsize=(9, 5))

    for species in species_list:

        species_data = df[df["species"] == species]

        ax.scatter(
            species_data[feature_x],
            species_data[feature_y],
            label=species_names.get(species, species),
            alpha=0.7,
            s=60
        )

    # 사용자가 입력한 데이터 표시
    ax.scatter(
        value_x,
        value_y,
        color="yellow",
        edgecolors="black",
        marker="*",
        s=350,
        label="입력한 꽃",
        zorder=5
    )

    ax.set_xlabel(feature_names[feature_x])
    ax.set_ylabel(feature_names[feature_y])

    ax.set_title("아이리스 꽃 품종 분포")

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "⭐ 노란색 별표는 사용자가 입력한 꽃의 위치입니다."
    )

    # --------------------------------------------------
    # 알고리즘 설명
    # --------------------------------------------------

    st.divider()

    st.subheader("📚 머신러닝 알고리즘 알아보기")

    with st.expander("1. KNN (K-최근접 이웃)"):

        st.write("""
        - 새로운 데이터와 가까운 주변 데이터 5개를 찾습니다.
        - 주변 데이터 중 가장 많은 품종을 선택합니다.
        - 가까운 이웃의 영향을 받는 알고리즘입니다.
        """)

    with st.expander("2. 의사결정나무 (Decision Tree)"):

        st.write("""
        - 데이터를 여러 기준으로 나누어 판단합니다.
        - 꽃잎 길이, 꽃잎 너비 등의 조건을 이용합니다.
        - 나무의 가지처럼 질문을 반복하며 품종을 결정합니다.
        """)

    with st.expander("3. 랜덤포레스트 (Random Forest)"):

        st.write("""
        - 여러 개의 의사결정나무를 만들어 예측합니다.
        - 각각의 나무가 예측한 결과를 종합합니다.
        - 여러 나무의 판단을 모아 최종 품종을 결정합니다.
        """)

    st.caption(
        "※ 표시된 예측 확률은 모델의 predict_proba 결과이며, "
        "실제 예측의 정확도를 보장하는 값은 아닙니다."
    )


# --------------------------------------------------
# 6. 탭 구성
# --------------------------------------------------

tab1, tab2 = st.tabs([
    "🌷 꽃잎 특성으로 예측하기",
    "🌼 원하는 특성 2개 선택하기"
])


# ==================================================
# TAB 1
# 꽃잎 길이와 꽃잎 너비로 예측
# ==================================================

with tab1:

    st.header("🌷 꽃잎 길이와 너비로 품종 예측")

    st.write(
        "꽃잎 길이와 꽃잎 너비를 조절해 보세요. "
        "3가지 알고리즘의 예측 결과를 비교할 수 있습니다."
    )

    col1, col2 = st.columns(2)

    with col1:

        petal_length = st.slider(
            "🌸 꽃잎 길이 (cm)",
            min_value=float(df["petal_length"].min()),
            max_value=float(df["petal_length"].max()),
            value=float(df["petal_length"].median()),
            step=0.1,
            key="tab1_petal_length"
        )

    with col2:

        petal_width = st.slider(
            "🌸 꽃잎 너비 (cm)",
            min_value=float(df["petal_width"].min()),
            max_value=float(df["petal_width"].max()),
            value=float(df["petal_width"].median()),
            step=0.1,
            key="tab1_petal_width"
        )

    st.info(
        f"입력한 꽃의 특성: 꽃잎 길이 {petal_length}cm / "
        f"꽃잎 너비 {petal_width}cm"
    )

    run_prediction(
        "petal_length",
        "petal_width",
        petal_length,
        petal_width
    )


# ==================================================
# TAB 2
# 4가지 특성 중 2개 선택
# ==================================================

with tab2:

    st.header("🌼 원하는 특성 2개 선택하기")

    st.write(
        "꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비 중 "
        "원하는 특성 2개를 선택해 품종을 예측해 보세요."
    )

    col1, col2 = st.columns(2)

    with col1:

        feature_x = st.selectbox(
            "📌 가로축 특성 선택",
            options=feature_columns,
            format_func=lambda x: feature_names[x],
            index=0,
            key="tab2_feature_x"
        )

    # 가로축과 다른 특성만 선택 가능
    available_y_features = [
        feature for feature in feature_columns
        if feature != feature_x
    ]

    with col2:

        feature_y = st.selectbox(
            "📌 세로축 특성 선택",
            options=available_y_features,
            format_func=lambda x: feature_names[x],
            index=0,
            key="tab2_feature_y"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        value_x = st.slider(
            f"🌸 {feature_names[feature_x]} 입력",
            min_value=float(df[feature_x].min()),
            max_value=float(df[feature_x].max()),
            value=float(df[feature_x].median()),
            step=0.1,
            key="tab2_value_x"
        )

    with col2:

        value_y = st.slider(
            f"🌸 {feature_names[feature_y]} 입력",
            min_value=float(df[feature_y].min()),
            max_value=float(df[feature_y].max()),
            value=float(df[feature_y].median()),
            step=0.1,
            key="tab2_value_y"
        )

    st.info(
        f"선택한 특성: {feature_names[feature_x]} "
        f"{value_x} / {feature_names[feature_y]} {value_y}"
    )

    run_prediction(
        feature_x,
        feature_y,
        value_x,
        value_y
    )


# --------------------------------------------------
# 7. 데이터 미리보기
# --------------------------------------------------

st.divider()

with st.expander("📋 아이리스 데이터 확인하기"):

    st.write(f"전체 데이터 개수: {len(df)}개")

    st.dataframe(
        df.rename(columns=feature_names),
        use_container_width=True
    )

    st.write("품종별 데이터 개수")

    st.dataframe(
        df["species"].value_counts().rename("개수"),
        use_container_width=True
    )

st.caption("🌸 아이리스 머신러닝 품종 예측 프로그램")
