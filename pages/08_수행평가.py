
import streamlit as st
import pandas as pd


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="칼로리 맞춤 음식 추천",
    page_icon="🍽️",
    layout="wide"
)

# 디자인 CSS
st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #D97706;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 17px;
    color: #777777;
    margin-bottom: 25px;
}

.food-name {
    font-size: 22px;
    font-weight: bold;
    color: #333333;
}

.food-kcal {
    font-size: 30px;
    font-weight: bold;
    color: #16A34A;
}

.info-text {
    color: #777777;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 제목
# ==========================================

st.markdown(
    '<div class="main-title">🍽️ CALORIE MATCH</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    '내가 원하는음식을 찾아보세요!'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 3. CSV 파일 경로 설정
# ==========================================

# 현재 파이썬 파일은 pages 폴더에 있음
# CSV 파일은 pages의 상위 폴더에 있음

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILES = [
    BASE_DIR / "FOOD-DATA-GROUP1.csv",
    BASE_DIR / "FOOD-DATA-GROUP2.csv",
    BASE_DIR / "FOOD-DATA-GROUP3.csv",
    BASE_DIR / "FOOD-DATA-GROUP4.csv",
    BASE_DIR / "FOOD-DATA-GROUP5.csv"
]


# ==========================================
# 4. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    all_data = []

    for file_path in CSV_FILES:

        if not file_path.exists():
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {file_path.name}"
            )

        # 한글 인코딩 대응
        df = None

        for encoding in ["utf-8-sig", "utf-8", "cp949"]:

            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding
                )
                break

            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(
                f"{file_path.name} 파일의 인코딩을 확인해 주세요."
            )

        all_data.append(df)

    # 5개 파일 합치기
    data = pd.concat(
        all_data,
        ignore_index=True
    )

    # 필요한 열 확인
    required_columns = [
        "food",
        "Caloric Value"
    ]

    for col in required_columns:

        if col not in data.columns:
            raise ValueError(
                f"필요한 데이터 열이 없습니다: {col}"
            )

    # 숫자형 데이터 변환
    numeric_columns = [
        "Caloric Value",
        "Protein",
        "Fat",
        "Carbohydrates",
        "Sugars",
        "Dietary Fiber"
    ]

    for col in numeric_columns:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    # 음식 정리
    data["food"] = (
        data["food"]
        .astype("string")
        .str.strip()
    )

    # 음식 이름과 칼로리가 없는 데이터 제거
    data = data.dropna(
        subset=["food", "Caloric Value"]
    )

    data = data[
        data["food"].str.len() > 0
    ]

    # 칼로리가 0 이하인 데이터 제거
    data = data[
        data["Caloric Value"] > 0
    ].copy()

    # 중복 음식 제거
    data = data.drop_duplicates(
        subset=["food"]
    )

    return data


# ==========================================
# 5. 데이터 불러오기 오류 처리
# ==========================================

try:

    df = load_data()

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")

    st.write(e)

    st.info("""
    확인 사항

    1. CSV 파일 5개가 모두 업로드되었는지 확인하세요.
    2. CSV 파일이 pages 폴더의 상위 폴더에 있는지 확인하세요.
    3. 파일 이름이 코드와 정확히 일치하는지 확인하세요.
    """)

    st.stop()


# ==========================================
# 6. 데이터 소개
# ==========================================

st.subheader("📊 음식 데이터 소개")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "사용한 음식 데이터",
        f"{len(df):,}개"
    )

with col2:

    st.metric(
        "사용한 CSV 파일",
        "5개"
    )

with col3:

    st.metric(
        "영양정보",
        "6가지 주요 항목"
    )

st.divider()


# ==========================================
# 7. 목표 칼로리 선택
# ==========================================

st.subheader("🎯 원하는 칼로리를 선택하세요!")

st.write(
    "슬라이더를 움직여 원하는 칼로리를 선택해 보세요."
)

target_kcal = st.slider(
    "목표 칼로리 (kcal)",
    min_value=50,
    max_value=1500,
    value=400,
    step=25
)

st.markdown(
    f"""
    <div style="
        background-color:#FFF7ED;
        padding:20px;
        border-radius:12px;
        text-align:center;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <div style="font-size:16px;color:#92400E;">
            내가 선택한 목표 칼로리
        </div>
        <div style="
            font-size:38px;
            font-weight:bold;
            color:#D97706;
        ">
            {target_kcal:,} kcal
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 8. 칼로리 차이 계산
# ==========================================

# 목표 칼로리와 각 음식 칼로리의 차이 계산

df["칼로리 차이"] = (
    df["Caloric Value"] - target_kcal
).abs()

# 칼로리 차이가 작은 순서로 정렬

recommendations = df.sort_values(
    by=["칼로리 차이", "food"],
    ascending=[True, True]
).head(5)


# ==========================================
# 9. 음식 추천 결과
# ==========================================

st.subheader("🥗 나에게 맞는 추천 음식 TOP 5")

st.write(
    "목표 칼로리와 가장 가까운 음식 5개를 찾았습니다!"
)

if len(recommendations) == 0:

    st.warning("추천할 음식이 없습니다.")

else:

    for rank, (_, food) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        food_name = html.escape(
            str(food["food"])
        )

        kcal = float(food["Caloric Value"])

        difference = abs(
            kcal - target_kcal
        )

        with st.container(border=True):

            col1, col2 = st.columns([3, 1])

            with col1:

                st.markdown(
                    f'<div class="food-name">'
                    f'{rank}. 🍴 {food_name}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="food-kcal">'
                    f'{kcal:,.0f} kcal'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    f"목표 칼로리와 {difference:,.0f} kcal 차이"
                )

            with col2:

                st.metric(
                    "칼로리 차이",
                    f"{difference:,.0f} kcal"
                )

            # 영양정보 표시

            nutrition_columns = [
                ("Protein", "단백질"),
                ("Fat", "지방"),
                ("Carbohydrates", "탄수화물"),
                ("Sugars", "당류"),
                ("Dietary Fiber", "식이섬유")
            ]

            available_nutrition = []

            for column, label in nutrition_columns:

                if (
                    column in df.columns
                    and pd.notna(food.get(column))
                ):

                    available_nutrition.append(
                        (column, label)
                    )

            if available_nutrition:

                metric_cols = st.columns(
                    len(available_nutrition)
                )

                for metric_col, (
                    column,
                    label
                ) in zip(
                    metric_cols,
                    available_nutrition
                ):

                    value = float(food[column])

                    metric_col.metric(
                        label,
                        f"{value:,.1f} g"
                    )


# ==========================================
# 10. 데이터 분석 설명
# ==========================================

st.divider()

st.subheader("🔍 음식 추천은 어떻게 이루어질까요?")

st.markdown("""
이 웹페이지는 판다스(Pandas)를 이용하여
음식의 칼로리를 비교하는 데이터 분석 프로그램입니다.

**① 음식 데이터 수집**

5개의 CSV 파일에 저장된 음식 데이터를 하나로 합칩니다.

**② 목표 칼로리 입력**

사용자가 원하는 칼로리를 선택합니다.

**③ 칼로리 차이 계산**

각 음식의 칼로리와 목표 칼로리의 차이를 계산합니다.

**④ 음식 추천**

목표 칼로리와 차이가 가장 작은 음식 5개를 추천합니다.
""")

st.info("""
📌 참고 사항

음식의 칼로리와 영양소는 원본 데이터에 기록된 값을 사용합니다.
원본 데이터의 정확한 제공량 기준은 확인되지 않았으므로
실제 한 끼 섭취량의 칼로리라고 단정할 수 없습니다.
""")

st.caption(
    "CALORIE MATCH | 나에게 맞는 음식 찾기 🍽️"
)

