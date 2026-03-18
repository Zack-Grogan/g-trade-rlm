# g-trade-rlm

RLM service for AI-generated reports, hypotheses, conclusions, chat analysis, and graph-oriented lineage over G-Trade data. This service is advisory only: it does not control the trader and it does not own execution logic.

## Surfaces

- `POST /feedback/cycle` - generate follow-up hypothesis feedback
- `POST /reports/generate` - generate report bundles
- `POST /conclusions/submit` - persist operator conclusions
- `POST /chat/analyze` - advisory chat analysis for the operator console and MCP
- `GET /health` / `GET /config` - service health and configuration summary

## Auth and env

- **Auth:** `Authorization: Bearer <RLM_AUTH_TOKEN>` preferred. `GTRADE_INTERNAL_API_TOKEN` is also accepted for internal service-to-service access.
- **Required env:** `DATABASE_URL`, `OPENROUTER_API_KEY`
- **Recommended env:** `RLM_AUTH_TOKEN`, `GTRADE_INTERNAL_API_TOKEN`, `RLM_AI_PROVIDER`, `RLM_AI_MODEL`, `QSTASH_TOKEN`, `RLM_REPLAY_WORKER_URL`
- **Optional env:** Redis cache vars, Daytona vars, and any provider-specific AI settings

## Storage

- Core similarity and embedding retrieval now target Postgres / `pgvector`.
- Upstash Vector is not part of the critical retrieval path.

## Role in the system

- `analytics` stays read-only.
- `rlm` owns AI/report/hypothesis/conclusion writes.
- `mcp` and `web` call this service for advisory generation only.
