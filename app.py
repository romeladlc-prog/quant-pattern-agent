"""Local viewer for normalized Alpaca market bars."""
from datetime import date, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.alpaca_client import download_historical_bars


st.set_page_config(page_title="Visor de mercado", layout="wide")
st.title("Visor de datos de mercado")

with st.sidebar:
    ticker = st.text_input("Ticker", value="ARM").strip().upper()
    timeframe = st.selectbox("Timeframe", ["15Min", "1Hour", "4Hour", "1Day", "1Week"], index=3)
    session = (
        st.selectbox("Sesión", ["regular", "extended"])
        if timeframe in {"15Min", "1Hour", "4Hour"}
        else "regular"
    )
    if timeframe in {"1Day", "1Week"}:
        st.caption("Sesión: regular")
    start_date = st.date_input("Fecha inicial", value=date(2025, 1, 1))
    end_date = st.date_input("Fecha final", value=date(2025, 1, 31))
    submitted = st.button("Cargar datos")

if submitted:
    if not ticker:
        st.error("Ingresa un ticker.")
        st.stop()
    if start_date > end_date:
        st.error("La fecha inicial debe ser anterior o igual a la fecha final.")
        st.stop()

    try:
        with st.spinner("Descargando datos normalizados..."):
            # The client uses [start, end); the selected final calendar day is inclusive.
            bars = download_historical_bars(
                symbol=ticker,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                timeframe=timeframe,
                session=session,
            )
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
        st.stop()

    st.metric("Total de filas", len(bars))
    info = st.columns(4)
    info[0].metric("Primera fecha (UTC)", str(bars["timestamp"].iloc[0]) if not bars.empty else "—")
    info[1].metric("Última fecha (UTC)", str(bars["timestamp"].iloc[-1]) if not bars.empty else "—")
    info[2].metric("Sesión", session)
    info[3].metric("Timeframe", timeframe)

    if timeframe == "4Hour" and not bars.empty:
        complete = int(bars["is_complete"].sum())
        st.info(f"Bloques completos: {complete} · Bloques parciales: {len(bars) - complete}")

    if bars.empty:
        st.warning("No hay datos para la selección.")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.04)
        fig.add_trace(go.Candlestick(
            x=bars["timestamp"], open=bars["open"], high=bars["high"],
            low=bars["low"], close=bars["close"], name="OHLC",
        ), row=1, col=1)
        fig.add_trace(go.Bar(x=bars["timestamp"], y=bars["volume"], name="Volumen"),
                      row=2, col=1)
        fig.update_layout(title=f"{ticker} · {timeframe} · {session}",
                          height=700, xaxis_rangeslider_visible=False,
                          margin=dict(l=20, r=20, t=60, b=20))
        fig.update_yaxes(title_text="Precio", row=1, col=1)
        fig.update_yaxes(title_text="Volumen", row=2, col=1)
        st.plotly_chart(fig, width="stretch")
