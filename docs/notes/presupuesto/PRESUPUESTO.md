# Presupuesto del proyecto

Calculado por `scripts/presupuesto.py` a partir de `docs/notes/presupuesto/bases.json` (bases verificadas el 2026-11-13). Redondeo: mitad hacia arriba a centimos en cada paso, y los pasos siguientes se calculan con el valor redondeado.

| Concepto | Base de calculo | Importe (EUR) |
|---|---|---:|
| Personal (autor) | 300 h x 13.39 EUR/h (24105.37 EUR / 1800 h) | 4 017.00 |
| Portatil (amortizacion) | 2000 EUR x 0.25 = 500.00 EUR/anio; x 300/1800 h | 83.33 |
| API de modelos (OpenRouter) | 6.34 USD / 1.1081 USD por EUR | 5.72 |
| Suscripcion a asistente de IA | 5 meses x 20 USD = 100.00 USD / 1.1081 | 90.24 |
| Software libre y planes gratuitos | Python, FastAPI, Neon, Vercel, TinyTeX | 0.00 |
| **Total** | | **4 196.29** |

No incluye las horas del tutor ni las cuotas patronales de la Seguridad Social.

## Fuentes

- **horas**: 300 (horas). Carga nominal del TFG: 12 creditos, "entre las 300 y 360 horas de trabajo efectivo" (guia de la escuela, .tfg-reference/text/Recomendaciones-Escritura-Defensa-TFG-7_11_18.txt, linea 8). Se usa el limite inferior. El autor no llevo registro de horas.
- **salario_anual**: 24105.37 (EUR por anio (bruto, sin cuota patronal)). XIX Convenio colectivo estatal de empresas de consultoria, tecnologias de la informacion y estudios de mercado y de la opinion publica. Tablas salariales actualizadas: Resolucion de 13 de abril de 2026, BOE num. 100, 24 de abril de 2026, BOE-A-2026-9024 (https://www.boe.es/boe/dias/2026/04/24/pdfs/BOE-A-2026-9024.pdf). Area 3, Grupo C, Nivel 3, total 2026 = salario base 22136,89 + plus convenio 1968,48.
- **horas_anuales**: 1800 (horas por anio). Jornada maxima anual del articulo 20.1 del convenio, BOE num. 92, 16 de abril de 2025, BOE-A-2025-7766 (https://www.boe.es/boe/dias/2025/04/16/pdfs/BOE-A-2025-7766.pdf).
- **portatil_precio**: 2000 (EUR). Dato del autor: MacBook comprado en 2024, precio aproximado.
- **portatil_amortizacion**: 0.25 (fraccion anual (lineal)). Coeficiente maximo para "Equipos para procesos de informacion" en la tabla del articulo 12.1.a de la Ley 27/2014 del Impuesto sobre Sociedades (https://www.boe.es/buscar/act.php?id=BOE-A-2014-12328).
- **openrouter_csv**: docs/notes/presupuesto/openrouter_usage_2026-10-14_2026-11-13.csv (ruta del fichero (USD)). Exportacion de uso de OpenRouter de la clave del proyecto, del 2026-10-14 al 2026-11-13; columnas API Key, Model, Value, % of Total; 16 modelos.
- **suscripcion_mensual**: 20 (USD por mes). Dato del autor: suscripcion a un asistente de IA generativa (Claude) usado en el desarrollo y la redaccion.
- **suscripcion_meses**: 2025-12, 2026-01, 2026-02, 2026-10, 2026-11 (meses). Meses con actividad en el historial del repositorio: git log --all --format=%ad --date=format:%Y-%m | sort | uniq -c. Segun el autor (2026-11-13), mantiene la suscripcion de forma continua y la usa tambien fuera del trabajo; por eso solo se imputan los meses con actividad.
- **software_libre**: 0 (EUR). Python, FastAPI, PostgreSQL en el plan gratuito de Neon, plan gratuito de Vercel y TinyTeX (requirements.txt y docs/notes/2026-11-11-propuesta-nube-y-mcp.md).
- **tipo_cambio**: 1.1081 (USD por 1 EUR). Tipo de cambio de referencia del euro del Banco Central Europeo del 12 de noviembre de 2026 (https://www.ecb.europa.eu/stats/exchange/eurofxref/shared/pdf/2026/11/20261112.pdf).
