# V28 · Presupuesto del proyecto

- **Fecha:** 2026-11-13
- **Estado:** hecha; queda por confirmar la moneda del precio del portátil
- **Capa:** documentación (`docs/notes/presupuesto/`) y un script (`scripts/presupuesto.py`)
- **Commits:** sin commit todavía

## Resultado

Presupuesto total: **4 196.29 EUR**, calculado por el script a partir de las bases de `bases.json`.

| Concepto | Base de cálculo | Importe (EUR) |
|---|---|---:|
| Personal (autor) | 300 h x 13.39 EUR/h (24105.37 EUR / 1800 h) | 4 017.00 |
| Portátil (amortización) | 2000 EUR x 0.25 = 500.00 EUR/año; x 300/1800 h | 83.33 |
| API de modelos (OpenRouter) | 6.34 USD / 1.1081 USD por EUR | 5.72 |
| Suscripción a asistente de IA | 5 meses x 20 USD = 100.00 USD / 1.1081 | 90.24 |
| Software libre y planes gratuitos | Python, FastAPI, Neon, Vercel, TinyTeX | 0.00 |
| **Total** | | **4 196.29** |

## Qué se ha hecho

- `docs/notes/presupuesto/bases.json`: cada base con su valor, unidad y fuente (documento, identificador, URL, ruta del repositorio o «dato del autor»).
- `docs/notes/presupuesto/openrouter_usage_2026-10-14_2026-11-13.csv`: copia sin cambios de la exportación de uso de la clave del proyecto (16 modelos, 6.339355 USD). El script lee el total de este fichero.
- `scripts/presupuesto.py`: lee las bases y el CSV, calcula las partidas con `decimal.Decimal`, imprime la tabla y escribe `docs/notes/presupuesto/PRESUPUESTO.md`. Solo biblioteca estándar.
- `tests/evaluation/test_presupuesto.py`: cifras, suma del CSV, base ausente y escritura del informe (solo en rutas temporales).
- Redondeo: mitad hacia arriba a céntimos en cada paso impreso, y cada paso siguiente se calcula desde el valor redondeado, para que quien use una calculadora obtenga los mismos números.

## Por qué

La memoria necesita un presupuesto cuyas cifras se puedan rastrear hasta un fichero. Con las bases en JSON y un script corto, un lector puede cambiar una base (el tipo de cambio, las horas) y volver a obtener el total sin tocar el texto.

## Objetivo

Un presupuesto pequeño, reproducible y verificable: cada importe sale de una base con fuente, y el total se regenera con un comando.

## Bases y fuentes

- **Horas (300).** El autor no llevó registro de horas. La base es la carga nominal del TFG (12 créditos, «entre las 300 y 360 horas de trabajo efectivo»; guía de la escuela, `.tfg-reference/text/Recomendaciones-Escritura-Defensa-TFG-7_11_18.txt`, línea 8); se usa el límite inferior.
- **Coste por hora.** XIX Convenio colectivo estatal de empresas de consultoría, tecnologías de la información y estudios de mercado y de la opinión pública; tablas actualizadas por la Resolución de 13 de abril de 2026 (BOE núm. 100, 24 de abril de 2026, BOE-A-2026-9024): Área 3, Grupo C, Nivel 3, 24 105.37 EUR al año. Jornada máxima anual de 1 800 horas (artículo 20.1, BOE-A-2025-7766). Salario bruto, sin cuotas patronales.
- **Portátil.** MacBook de 2024, unos 2 000 EUR (dato del autor), amortización lineal al 25 % anual (máximo para «Equipos para procesos de información», artículo 12.1.a de la Ley 27/2014), con la parte de 300 horas sobre 1 800.
- **Software.** Gratuito (Python, FastAPI, PostgreSQL en el plan gratuito de Neon, plan gratuito de Vercel, TinyTeX). Una partida de pago: la suscripción al asistente de IA, 20 USD al mes (dato del autor), durante los cinco meses con actividad en el historial del repositorio (2025-12, 2026-01, 2026-02, 2026-10, 2026-11; comprobados el 2026-11-13 con `git log --all --format=%ad --date=format:%Y-%m | sort | uniq -c`).
- **Tipo de cambio.** 1 EUR = 1.1081 USD, referencia del Banco Central Europeo del 12 de noviembre de 2026.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Horas medidas | No existen: el autor no llevó registro. Inventar un reparto de horas daría una precisión falsa; la carga nominal del TFG es un dato citable. |
| Sueldo de mercado tomado de una encuesta salarial | Depende de la encuesta, del año y del puesto, y es difícil de citar con precisión. El convenio es un documento oficial con tablas por grupo y nivel. |
| Amortizar el portátil en el periodo máximo legal de 8 años | El 25 % anual es el coeficiente máximo de la tabla para equipos informáticos (periodo mínimo de 4 años) y refleja mejor la vida útil de un portátil. Con 8 años el importe sería la mitad. |
| Imputar el portátil por meses | El proyecto no ocupó meses completos de forma uniforme; imputar por horas usa la misma base que el personal (300 de 1 800) y no depende de un calendario. |

## Cómo reproducirlo

```
.venv/bin/python -m scripts.presupuesto
.venv/bin/python -m pytest tests/evaluation/test_presupuesto.py
```

Sin red. Imprime la tabla y reescribe `docs/notes/presupuesto/PRESUPUESTO.md`.

## Pendiente

- **El autor debe confirmar** el precio y la moneda del portátil (unos 2 000 EUR, aproximado), y el precio de la suscripción (20 USD al mes). El 2026-11-13 el autor indicó que mantiene la suscripción de forma continua y la usa también fuera del trabajo; por eso solo se imputan los cinco meses con actividad en el repositorio.
- **Horas del tutor:** no están incluidas.
- **Cuotas patronales** de la Seguridad Social: no incluidas (salario bruto).
- Las horas son nominales, no medidas; la memoria debe decirlo.

## Ficheros

- `docs/notes/presupuesto/bases.json`, `docs/notes/presupuesto/openrouter_usage_2026-10-14_2026-11-13.csv`, `docs/notes/presupuesto/PRESUPUESTO.md`
- `scripts/presupuesto.py`
- `tests/evaluation/test_presupuesto.py`
