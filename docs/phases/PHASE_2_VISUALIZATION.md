# Fase 2 — Visualization

**Estado:** completada como visor local. `app.py` usa Streamlit y Plotly para mostrar barras NORMALIZED de un ticker y rango elegidos por el usuario. Permite los marcos temporales y sesiones admitidos por Data Layer; 1Day y 1Week usan sesión regular. Se ejecuta con `python -m streamlit run app.py` desde el entorno Python 3.12.

La interfaz es un explorador de datos. No integra aún modelos, contexto externo, scanner ni señales. Requiere credenciales Alpaca en `.env` y acceso de red. `main.py` conserva una consulta mínima de ARM para inspección por consola.
