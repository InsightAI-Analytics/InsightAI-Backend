# InsightAI — Backend Service

FastAPI-powered analytical Q&A backend for structured datasets (CSV, XLSX, XLS). InsightAI uses an LLM (Google Gemini or OpenAI-compatible) strictly to parse user intent into a structured JSON schema, while delegating data transformation, filtering, mathematical operations, relational joins, and statistical delta analysis deterministically to Pandas and a custom Delta Solutioning engine.

---

## Architecture & Request Flow

```
                                      ┌───────────────────────────────────────────┐
                                      │       Client (Web App / API Consumer)     │
                                      └─────────────────────┬─────────────────────┘
                                                            │
                     ┌──────────────────────────────────────┴──────────────────────────────────────┐
                     │ POST /api/files/upload (multipart)                                          │ POST /api/query (JSON)
                     ▼                                                                             ▼
       ┌───────────────────────────┐                                                 ┌───────────────────────────┐
       │   Parser (services/parser)│                                                 │  LLM Intent Extraction    │
       │   - Encodings & Delimiters│                                                 │  (services/llm_service)   │
       │   - Excel/openpyxl engine │                                                 │  - Gemini (default)       │
       └─────────────┬─────────────┘                                                 │  - OpenAI fallback        │
                     ▼                                                               └─────────────┬─────────────┘
       ┌───────────────────────────┐                                                               │ Structured QueryIntent
       │Normalizer (services/norm) │                                                               ▼
       │   - snake_case columns    │                                                 ┌───────────────────────────┐
       │   - Type inference        │                                                 │  Pandas Analytics Engine  │
       └─────────────┬─────────────┘                                                 │   (services/analytics)    │
                     ▼                                                               │   - Auto Relational Merge │
       ┌───────────────────────────┐                                                 │   - GroupBy, Aggregations │
       │  In-Memory Session Store  │◄──────────────── Dataset Lookup ────────────────┤   - Time-series Trends    │
       │   (session_store.py)      │                                                 └─────────────┬─────────────┘
       └───────────────────────────┘                                                               │ Raw Results + ChartData
                                                                                                   ▼
                                                                                     ┌───────────────────────────┐
                                                                                     │ Delta Solutioning Engine  │
                                                                                     │ (services/delta_solution) │
                                                                                     │ - Period-over-period %    │
                                                                                     │ - Anomaly / Spread        │
                                                                                     │ - Prescriptive Actions    │
                                                                                     │ - Next Drill-down Queries │
                                                                                     └─────────────┬─────────────┘
                                                                                                   ▼
                                                                                     ┌───────────────────────────┐
                                                                                     │ Response Builder & Schema │
                                                                                     │ (services/response_builder│
                                                                                     └───────────────────────────┘
```

### Analytical Separation of Concerns
1. **Zero LLM Arithmetic**: LLMs struggle with precise arithmetic across large datasets. The LLM is restricted to semantic understanding—extracting entities, metrics, dimensions, filters, chart requirements, and relational join keys.
2. **Deterministic Computation**: All aggregations (`sum`, `average`, `count`, `min`, `max`), comparisons, and temporal trends are calculated using vectorized Pandas operations.
3. **Delta Solutioning Layer**: On top of baseline math, an automated analytical layer detects anomalies, computes period-over-period percentage variance, identifies top contributor concentration, and prescribes actionable recommendations with interactive follow-up drill-downs.

---

## Tech Stack & Core Dependencies

- **Runtime**: Python 3.10+
- **Web Framework**: FastAPI `0.115.0`
- **ASGI Server**: Uvicorn `0.30.6`
- **Data Engine**: Pandas `2.2.3`, NumPy, OpenPyXL `3.1.5`
- **Validation**: Pydantic `2.9.2`
- **LLM Integrations**:
  - Google GenAI SDK (`google-genai >= 1.73.0`)
  - OpenAI Python SDK (`openai == 1.51.0`)
- **Testing**: Pytest `8.3.3`, Pytest-Asyncio `0.24.0`, HTTPX

---

## Project Structure

```
InsightAI-backend/
├── main.py                    # FastAPI entrypoint, CORS configuration & health checks
├── config.py                  # Pydantic/dotenv application settings
├── session_store.py           # Thread-safe in-memory dataset session registry
├── smoke_test.py              # End-to-end integration smoke script
├── requirements.txt           # Production and testing Python dependencies
├── pytest.ini                 # Pytest runner configuration
├── .env.example               # Environment variable templates
├── models/
│   └── schemas.py             # Pydantic models (QueryIntent, DeltaSolution, etc.)
├── routers/
│   ├── files.py               # Upload, list, and delete endpoints
│   └── query.py               # Query execution endpoint
├── services/
│   ├── parser.py              # Multi-encoding CSV & Excel stream parser
│   ├── normalizer.py          # Column snake_case normalization and type inference
│   ├── llm_service.py         # Google Gemini & OpenAI structured intent extractor
│   ├── analytics.py           # Vectorized Pandas query and relational merge engine
│   ├── delta_solutioning.py   # Variance, anomaly detection, recommendations & follow-up questions
│   └── response_builder.py    # Formatter mapping analytics results to natural language
└── tests/
    ├── test_parser.py         # Parsing and normalization test suite
    └── test_analytics.py      # Analytics, relational joins, and delta tests
```

---

## Environment Variables

All settings load from environment variables or a local `.env` file via [config.py](config.py):

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (if using Gemini) | `""` | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | Optional fallback | `""` | OpenAI API key (used if `GEMINI_API_KEY` is not set) |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Custom OpenAI-compatible endpoint |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model name |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000` | Comma-separated list of permitted CORS origins |
| `MAX_FILE_SIZE_MB` | No | `50` | Maximum file upload size limit per file in MB |
| `MAX_FILES_PER_SESSION`| No | `10` | Maximum number of uploaded files allowed per session |

---

## API Reference

### 1. Health Checks
- **`GET /`** & **`GET /health`**
  - **Purpose**: Service liveness and metadata verification.
  - **Response (`200 OK`)**:
    ```json
    { "status": "ok" }
    ```

### 2. Upload Files
- **`POST /api/files/upload`**
  - **Purpose**: Uploads and parses one or more CSV or Excel files into a session.
  - **Headers**: `X-Session-ID: <string>` (Required UUID)
  - **Body**: `multipart/form-data` with one or more files in the `files` field.
  - **Response (`200 OK`)**:
    ```json
    {
      "session_id": "session-uuid",
      "uploaded": [
        {
          "name": "sales.csv",
          "rows": 100,
          "columns": 4,
          "column_info": [
            {"name": "order_date", "original_name": "Order Date", "dtype": "datetime64[ns]"},
            {"name": "revenue", "original_name": "Revenue ($)", "dtype": "float64"}
          ],
          "sample": [{"order_date": "2024-01-01", "revenue": 1250.0}]
        }
      ],
      "errors": []
    }
    ```

### 3. List Files
- **`GET /api/files`**
  - **Purpose**: Retrieves metadata, schema, and samples for all datasets loaded in the session.
  - **Headers**: `X-Session-ID: <string>` (Required UUID)
  - **Response (`200 OK`)**: Array of `DatasetInfo` objects.

### 4. Delete File
- **`DELETE /api/files/{filename}`**
  - **Purpose**: Evicts a specific dataset from the session.
  - **Headers**: `X-Session-ID: <string>` (Required UUID)
  - **Response (`200 OK`)**:
    ```json
    { "message": "File 'sales.csv' removed successfully." }
    ```

### 5. Natural Language Query
- **`POST /api/query`**
  - **Purpose**: Processes a natural language question across the session datasets.
  - **Request Body (`application/json`)**:
    ```json
    {
      "session_id": "session-uuid",
      "question": "What is the total revenue by region?"
    }
    ```
  - **Response (`200 OK`)**:
    ```json
    {
      "question": "What is the total revenue by region?",
      "answer": "Here is the breakdown of **Revenue** by **Region**:\n• North: 45,000\n• South: 32,000",
      "metrics": {
        "operation": "sum",
        "metric": "revenue",
        "group_by": "region",
        "values": {"North": 45000.0, "South": 32000.0},
        "total_groups": 2
      },
      "chart": {
        "chart_type": "bar",
        "labels": ["North", "South"],
        "datasets": [{"label": "Revenue", "data": [45000.0, 32000.0]}],
        "x_label": "Region",
        "y_label": "Revenue"
      },
      "delta_solution": {
        "insights": [
          {
            "metric": "Top Contributor: North",
            "change_percent": 58.4,
            "direction": "up",
            "summary": "'North' accounts for 58.4% of overall Revenue (45,000.00)."
          }
        ],
        "recommendations": [
          "Double down on strategies in 'North' while conducting a gap analysis on lower performing segment 'South'."
        ],
        "suggested_questions": [
          "Show details for top Region 'North'",
          "Why is Region 'South' underperforming in Revenue?"
        ]
      },
      "error": null
    }
    ```

---

## Data Processing & Cross-File Relational Joins

1. **Robust File Parsing**:
   - Handles various CSV encodings (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`) and delimiters (`,`, `;`, `\t`, `|`).
   - Parses Excel (`.xlsx`, `.xls`) via `openpyxl`.
2. **Schema Normalization**:
   - Converts columns to snake_case, resolves collisions with numeric suffixes, and preserves original column mappings for rendering.
   - Strips currency symbols (`$`, `€`, `£`, `₹`), percent signs, and commas to infer numeric types.
   - Detects date/time columns automatically.
3. **Cross-File Relational Joins**:
   - Automatically detects shared identifier columns (e.g., `product_id`, `customer_id`, `sku`) or explicit `join_keys` extracted from user queries.
   - Executes an inner merge (`pd.merge`), disambiguating overlapping column names.
   - Gracefully falls back to `pd.concat` when schemas are identical or when records represent appended time batches.

---

## Local Setup & Run Instructions

### 1. Prerequisites
- Python 3.10+
- A valid Google Gemini API key or OpenAI-compatible key

### 2. Installation
```bash
cd InsightAI-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
copy .env.example .env
```
Open `.env` and configure:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
ALLOWED_ORIGINS=http://localhost:3000
```

### 4. Run Development Server
```bash
uvicorn main:app --reload --port 8000
```
- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### 5. Running Test Suite
Execute the full test suite (30 unit tests covering parser, normalizer, analytics, relational joins, and delta solutioning):
```bash
python -m pytest tests/ -v
```

---

## Technical Design Decisions

- **Two-Phase Separation**: Intent extraction is separated from computation. LLMs generate the formal analytical intent, while Pandas performs 100% of the mathematical calculations to prevent hallucinations.
- **Session-Based Isolation**: Sessions are identified via the `X-Session-ID` header. Datasets are stored per session in memory, protected by thread locks (`threading.Lock`).
- **Graceful Error Recovery**: If an LLM cannot parse a question or a filter produces an empty DataFrame, informative, user-friendly feedback is returned rather than unhandled 500 exceptions.

---

## Limitations & Realistic Future Improvements

- **In-Memory Volatility**: The current session store is in-memory (`session_store.py`). Server restarts clear active sessions. A production evolution would back this store with Redis or PostgreSQL.
- **Single Sheet Excel**: Currently parses the first sheet of uploaded workbooks. Multi-sheet selection can be added.
- **Streaming LLM Responses**: Future releases can stream intent generation and analytical synthesis via Server-Sent Events (SSE).
