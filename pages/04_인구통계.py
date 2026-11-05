# --------------------------------------------------
# 선택한 조건에 맞는 데이터 추출
# --------------------------------------------------
selected_row = df.loc[
    df["행정구역_표시"] == selected_district
].iloc[0]

gender_code = GENDER_LABELS[selected_gender]
prefix = f"{selected_year}년_{gender_code}_"

age_columns = [prefix + age for age in AGE_GROUPS]
missing_columns = [column for column in age_columns if column not in df.columns]

if missing_columns:
    st.error(
        "선택한 조건에 해당하는 데이터 열을 찾지 못했습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

population_values = selected_row[age_columns].fillna(0).astype(int).tolist()
total_population = sum(population_values)

chart_data = pd.DataFrame({
    "연령대": AGE_GROUPS,
    "인구수": population_values
})


# --------------------------------------------------
# 요약 지표
# --------------------------------------------------
st.divider()

metric1, metric2, metric3 = st.columns(3)
metric1.metric("선택한 행정구역", selected_district)
metric2.metric("선택한 연도", f"{selected_year}년")
metric3.metric("선택 조건의 인구 합계", f"{total_population:,}명")

# --------------------------------------------------
# 핫핑크 꺾은선 그래프
# --------------------------------------------------
st.subheader("📈 연령대별 인구수")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_data["연령대"],
        y=chart_data["인구수"],
        mode="lines+markers",
        name="인구수",
        line=dict(
            color=HOT_PINK,
            width=4
        ),
        marker=dict(
            color=HOT_PINK,
            size=9,
            line=dict(color="#FFFFFF", width=1)
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "인구수: %{y:,.0f}명"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title=dict(
        text=f"{selected_district} · {selected_gender} · {selected_year}년",
        x=0.02,
        xanchor="left"
    ),
    xaxis=dict(
        title="나이",
        type="category",
        categoryorder="array",
        categoryarray=AGE_GROUPS,
        tickmode="linear",
        dtick=1,
        showgrid=True,
        gridcolor="#F3B6D2",
        gridwidth=1,
        zeroline=False
    ),
    yaxis=dict(
        title="인구수(명)",
        rangemode="tozero",
        tickmode="linear",
        dtick=100,
        tickformat=",d",
        showgrid=True,
        gridcolor="#E8DCE2",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#B8AAB1"
    ),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(
        family="Arial, sans-serif",
        size=13,
        color="#333333"
    ),
    hovermode="x unified",
    height=540,
    margin=dict(l=30, r=25, t=75, b=65)
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displayModeBar": True, "responsive": True}
)

# --------------------------------------------------
# 선택 데이터 표
# --------------------------------------------------
with st.expander("📋 그래프에 사용된 인구 데이터 보기"):
    display_data = chart_data.copy()
    display_data["인구수"] = display_data["인구수"].map(lambda x: f"{x:,}명")
    st.dataframe(display_data, use_container_width=True, hide_index=True)

st.caption(
    "자료: 제공된 popuiation.csv | 연령대는 원본 자료의 10세 단위 구간을 사용합니다."
)
