# Roadmap

| Fase | Estado | Alcance |
|---|---|---|
| 1. Data Layer | Completada | RAW, NORMALIZED y validación temporal. |
| 2. Visualization | Completada | Visor Streamlit de barras. |
| 3. Feature Engine | Completada | Features estadísticas causales. |
| 4. Quant Models | Completada | Familias complementarias, sin señales. |
| 4.1 Stability | Completada | Walk-forward y audit cross-asset. |
| 5. External Context | Completada | VIX, liquidez, breadth y noticias, con límites de disponibilidad. |
| 6. Market Structure | Completada | Swings causales, estructura, niveles, eventos y validación multi-timeframe, sin señales. |
| 6.1 Hardening | Completada (tests sin red) | Niveles y breakouts as-of, recencia de breakout, RS sin relleno, semanas `is_complete`, feed único, metadata de modelos, redacción de claves, CI. Validación en vivo pendiente de credenciales. |
| 7. Pattern Engine | Completada (tests y validación sintética) | Patrones compuestos causales con máquina de estados, evidencia a favor/en contra y `convergence_score`; tabla de episodios para Fase 8. Validación con datos Alpaca pendiente de credenciales. |
| 8. Backtesting | Pendiente | Sin implementación. Consumirá `pattern_events.csv`; los episodios `is_open` son provisionales. |
| 9. Supervised ML | Pendiente | Sin implementación. |
| 10. Ensemble | Pendiente | Sin implementación. |
| 11. Obsidian Knowledge Layer | Pendiente | Sin integración implementada. |
| 12. Agent | Pendiente | Sin implementación. |
| 13. Scanner/Alerts | Pendiente | Sin implementación. |

"Completada" significa que existe la implementación de esa fase y se ejecutó su validación documentada; no implica utilidad predictiva, publicación ni preparación para operar.
