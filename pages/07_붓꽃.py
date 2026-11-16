
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="아이리스 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 아이리스 붓꽃 품종 예측")
st.write("꽃잎 길이와 너비를 입력하면 붓꽃의 품종을 예측합니다.")
st.divider()


# =====================================
# 2. 데이터 불러오기
# =====================================

@st.cache_data
def load_data():

    file_path = Path(__file__).parent / "아이리스.csv"

    if file_path.exists():

        df = pd.read_csv(file_path, encoding="cp949")

        required_columns = [
            "꽃잎 길이",
            "꽃잎 너비",
            "품종"
        ]

        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV 파일의 열 이름을 확인해주세요.")

        df = df[required_columns].copy()

    else:

        # CSV 파일이 없으면 sklearn 기본 데이터 사용
        iris = load_iris()

        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )

        df = df.rename(columns={
            "petal length (cm)": "꽃잎 길이",
            "petal width (cm)": "꽃잎 너비"
        })

        df["품종"] = [
            "Iris-" + iris.target_names[i]
            for i in iris.target
        ]

        df = df[
            ["꽃잎 길이", "꽃잎 너비", "품종"]
        ]

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# =====================================
# 3. 데이터 범위 확인
# =====================================

min_length = float(df["꽃잎 길이"].min())
max_length = float(df["꽃잎 길이"].max())

min_width = float(df["꽃잎 너비"].min())
max_width = float(df["꽃잎 너비"].max())


# =====================================
# 4. 머신러닝 모델 생성
# =====================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ))
])

X = df[["꽃잎 길이", "꽃잎 너비"]]
y = df["품종"]

model.fit(X, y)


# =====================================
# 5. 데이터 정보
# =====================================

st.subheader("📊 데이터 정보")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df)}개")
col2.metric("꽃잎 길이 범위",
            f"{min_length} ~ {max_length} cm")
col3.metric("꽃잎 너비 범위",
            f"{min_width} ~ {max_width} cm")

st.info(
    "예측 알고리즘: KNN (K=5) + StandardScaler"
)


# =====================================
# 6. 사용자 입력
# =====================================

st.subheader("🌷 꽃잎 크기 입력하기")

col1, col2 = st.columns(2)

with col1:

    petal_length = st.slider(
        "🌼 꽃잎 길이 (cm)",
        min_value=min_length,
        max_value=max_length,
        value=round(
            float(df["꽃잎 길이"].median()), 1
        ),
        step=0.1
    )

with col2:

    petal_width = st.slider(
        "🌼 꽃잎 너비 (cm)",
        min_value=min_width,
        max_value=max_width,
        value=round(
            float(df["꽃잎 너비"].median()), 1
        ),
        step=0.1
    )

st.caption(
    f"입력 가능한 범위: "
    f"꽃잎 길이 {min_length}~{max_length}cm / "
    f"꽃잎 너비 {min_width}~{max_width}cm"
)


# =====================================
# 7. 품종 예측
# =====================================

if st.button("🔍 붓꽃 품종 예측하기", use_container_width=True):

    input_data = pd.DataFrame(
        [[petal_length, petal_width]],
        columns=["꽃잎 길이", "꽃잎 너비"]
    )

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    korean_names = {
        "Iris-setosa": "세토사 (Setosa)",
        "Iris-versicolor": "버시컬러 (Versicolor)",
        "Iris-virginica": "버지니카 (Virginica)"
    }

    st.divider()

    st.subheader("🎉 예측 결과")

    st.success(
        f"예측된 붓꽃 품종은 "
        f"**{korean_names.get(prediction, prediction)}** 입니다!"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "꽃잎 길이",
            f"{petal_length} cm"
        )

    with col2:
        st.metric(
            "꽃잎 너비",
            f"{petal_width} cm"
        )

    # 품종별 예측 확률
    st.subheader("📈 품종별 예측 확률")

    probability_df = pd.DataFrame({
        "품종": [
            korean_names.get(name, name)
            for name in model.classes_
        ],
        "예측 확률": probabilities
    })

    probability_df["예측 확률"] *= 100

    for _, row in probability_df.iterrows():

        st.write(
            f"**{row['품종']} : "
            f"{row['예측 확률']:.1f}%**"
        )

        st.progress(
            float(row["예측 확률"] / 100)
        )


    # =====================================
    # 8. 산점도 시각화
    # =====================================

    st.divider()

    st.subheader("📍 꽃잎 크기 산점도")

    st.write(
        "가로축은 꽃잎 길이, 세로축은 꽃잎 너비입니다. "
        "노란색 별은 사용자가 입력한 값입니다."
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # 품종별 실제 데이터
    for species in sorted(df["품종"].unique()):

        species_data = df[df["품종"] == species]

        ax.scatter(
            species_data["꽃잎 길이"],
            species_data["꽃잎 너비"],
            label=species,
            s=55,
            alpha=0.7
        )

    # 사용자가 입력한 데이터: 노란색 별
    ax.scatter(
        petal_length,
        petal_width,
        color="yellow",
        edgecolors="black",
        marker="*",
        s=450,
        linewidths=1.5,
        label="Your Input",
        zorder=10
    )

    ax.set_xlabel("Petal Length (cm)", fontsize=12)
    ax.set_ylabel("Petal Width (cm)", fontsize=12)

    ax.set_title(
        "Iris Petal Length and Width",
        fontsize=15
    )

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "🌟 노란색 별: 사용자가 입력한 꽃잎 크기"
    )

    st.info(
        "산점도에서 노란색 별과 가까운 실제 데이터의 "
        "품종을 참고하여 KNN 알고리즘이 예측합니다."
    )

