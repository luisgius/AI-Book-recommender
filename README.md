# Sistema Inteligente de Recomendación de Libros mediante PLN y LLMs

TFG (Bachelor's Thesis) - Book Recommendation System

## Descripción

Sistema híbrido de búsqueda y recomendación de libros que combina:
- Búsqueda léxica (BM25)
- Búsqueda semántica (embeddings + FAISS)
- LLM para interpretación de consultas y generación de explicaciones

## Arquitectura

Hexagonal Architecture (Ports & Adapters):
- **Domain**: Entidades, servicios y puertos
- **Infrastructure**: Implementaciones de BD, búsqueda, LLM, etc.
- **API**: FastAPI REST endpoints
- **UI**: Interfaz web simple para testing

## Estructura del Proyecto

```
.
├── app/                    # Código principal
│   ├── api/               # API REST
│   ├── domain/            # Capa de dominio
│   ├── infrastructure/    # Adaptadores
│   ├── ingestion/         # Jobs de ingesta de datos
│   ├── evaluation/        # Evaluación del sistema
│   └── ui/                # Interfaz web
├── tests/                 # Tests unitarios e integración
├── data/                  # Datos, índices, BD
├── docs/                  # Documentación
├── scripts/               # Scripts auxiliares
├── requirements.txt       # Dependencias Python
└── Dockerfile            # Containerización
```

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar API
python -m app.main

# Ejecutar tests
pytest

# Ingestar libros
python -m app.ingestion.ingest_books_job

# Evaluar sistema
python -m app.evaluation.evaluation_job
```

## Tecnologías

- Python 3.11+
- FastAPI
- SQLite
- BM25
- FAISS
- LangChain
- pytest

## Autor

Luis Giménez - TFG 2025
