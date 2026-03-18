-- RLM schema: knowledge_store (meta-learner), trade_embeddings (semantic search), replay_runs (replay metadata)
-- Shares base tables (runs, completed_trades, events, state_snapshots) with analytics.

CREATE TABLE IF NOT EXISTS knowledge_store (
    id BIGSERIAL PRIMARY KEY,
    hypothesis_id TEXT NOT NULL,
    result_id BIGINT,
    verdict TEXT NOT NULL,
    confidence_score REAL,
    mutation_directive TEXT,
    regime_tags JSONB,
    survival_count INT DEFAULT 0,
    rejection_count INT DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_validated_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_knowledge_store_hypothesis ON knowledge_store(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_store_created ON knowledge_store(created_at DESC);

CREATE TABLE IF NOT EXISTS trade_embeddings (
    id BIGSERIAL PRIMARY KEY,
    trade_id BIGINT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(trade_id)
);
CREATE INDEX IF NOT EXISTS idx_trade_embeddings_trade_id ON trade_embeddings(trade_id);

CREATE TABLE IF NOT EXISTS replay_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    result_summary JSONB,
    what_if_config JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_replay_runs_run_id ON replay_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_replay_runs_status ON replay_runs(status);
