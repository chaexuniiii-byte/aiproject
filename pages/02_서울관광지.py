import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(
    page_title="서울 관광지 TOP 10",
    page_icon="🇰🇷",
    layout="wide"
)

st.title("🇰🇷 외국인 여행자를 위한 서울 관광지 TOP 10")
st.write("서울의 대표 관광 명소 10곳을 지도에서 만나보세요!")

# 관광지 데이터
places = [
    {"순위": 1, "관광지": "경복궁", "영문명": "Gyeongbokgung Palace",
     "위도": 37.5788, "경도": 126.9770,
     "소개": "조선 왕조의 대표 궁궐입니다. 전통 건축과 수문장 교대식을 볼 수 있으며 한복 체험도 인기가 많습니다.",
     "추천": "한국 전통문화, 한복 체험"},

    {"순위": 2, "관광지": "명동", "영문명": "Myeongdong",
     "위도": 37.5636, "경도": 126.9850,
     "소개": "쇼핑과 길거리 음식으로 유명합니다. K-뷰티 제품과 다양한 한국 간식을 즐길 수 있습니다.",
     "추천": "쇼핑, 길거리 음식, K-뷰티"},

    {"순위": 3, "관광지": "남산서울타워", "영문명": "N Seoul Tower",
     "위도": 37.5512, "경도": 126.9882,
     "소개": "서울의 아름다운 전경을 감상할 수 있는 전망 명소입니다. 특히 야경이 아름답습니다.",
     "추천": "서울 전망, 야경"},

    {"순위": 4, "관광지": "북촌한옥마을", "영문명": "Bukchon Hanok Village",
     "위도": 37.5826, "경도": 126.9830,
     "소개": "전통 한옥이 모여 있는 마을입니다. 아름다운 골목을 걸으며 옛 서울의 분위기를 느낄 수 있습니다.",
     "추천": "한옥, 전통문화, 사진 촬영"},

    {"순위": 5, "관광지": "인사동", "영문명": "Insadong",
     "위도": 37.5740, "경도": 126.9857,
     "소개": "전통 공예품과 갤러리, 찻집이 모여 있는 거리입니다. 한국적인 기념품을 구매하기 좋습니다.",
     "추천": "전통 공예, 기념품, 전통차"},

    {"순위": 6, "관광지": "홍대거리", "영문명": "Hongdae Street",
     "위도": 37.5563, "경도": 126.9236,
     "소개": "젊고 활기찬 분위기의 거리입니다. 개성 있는 카페와 패션 매장, 버스킹 공연을 즐길 수 있습니다.",
     "추천": "K-팝, 공연, 카페"},

    {"순위": 7, "관광지": "동대문디자인플라자",
     "영문명": "Dongdaemun Design Plaza",
     "위도": 37.5671, "경도": 127.0095,
     "소개": "독특한 곡선형 건축물이 인상적인 문화 공간입니다. 전시와 행사도 자주 열립니다.",
     "추천": "건축, 디자인, 전시"},

    {"순위": 8, "관광지": "광장시장", "영문명": "Gwangjang Market",
     "위도": 37.5700, "경도": 126.9997,
     "소개": "빈대떡, 김밥, 떡볶이 등 다양한 한국 음식을 맛볼 수 있는 전통시장입니다.",
     "추천": "한국 음식, 전통시장"},

    {"순위": 9, "관광지": "청계천", "영문명": "Cheonggyecheon Stream",
     "위도": 37.5692, "경도": 126.9788,
     "소개": "서울 도심 한가운데 흐르는 아름다운 산책로입니다. 물길을 따라 여유롭게 산책할 수 있습니다.",
     "추천": "산책, 휴식, 야경"},

    {"순위": 10, "관광지": "롯데월드", "영문명": "Lotte World",
     "위도": 37.5111, "경도": 127.0982,
     "소개": "다양한 놀이기구와 퍼레이드를 즐길 수 있는 테마파크입니다. 가족과 친구에게 추천합니다.",
     "추천": "놀이기구, 가족 여행"}
]

df = pd.DataFrame(places)

# 지도 제목
st.subheader("📍 서울 관광지 지도")
st.write("숫자 마커에 마우스를 올리면 관광지 정보를 확인할 수 있습니다.")

# 지도 마커
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[경도, 위도]",
    get_radius=170,
    get_fill_color=[230, 60, 60, 220],
    pickable=True
)

# 지도 숫자 표시
text_layer = pdk.Layer(
    "TextLayer",
    data=df,
    get_position="[경도, 위도]",
    get_text="순위",
    get_size=16,
    get_color=[255, 255, 255],
    get_alignment_baseline="'center'"
)

view = pdk.ViewState(
    latitude=37.5665,
    longitude=126.9780,
    zoom=10.7,
    pitch=0
)

deck = pdk.Deck(
    layers=[layer, text_layer],
    initial_view_state=view,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={
        "html": "<b>{순위}. {관광지}</b><br/>{영문명}<br/>{추천}",
        "style": {
            "backgroundColor": "#222",
            "color": "white"
        }
    }
)

st.pydeck_chart(deck, use_container_width=True, height=560)

# 관광지 설명
st.divider()
st.subheader("🧭 서울 관광지 상세 소개")
st.write("외국인 친구들에게 소개하고 싶은 서울의 명소를 살펴보세요!")

for i in range(0, len(places), 2):
    cols = st.columns(2)

    for col, place in zip(cols, places[i:i+2]):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"### {place['순위']}. {place['관광지']}"
                )
                st.caption(place["영문명"])
                st.write(place["소개"])
                st.markdown(
                    f"✨ **추천 포인트:** {place['추천']}"
                )

                url = (
                    "https://www.google.com/maps/search/?api=1"
                    f"&query={place['위도']},{place['경도']}"
                )

                st.link_button("📍 구글 지도에서 위치 확인", url)

st.divider()
st.caption(
    "※ 관광지 목록은 대표 명소를 선정한 안내용 목록이며 "
    "공식 실시간 인기 순위는 아닙니다."
)
