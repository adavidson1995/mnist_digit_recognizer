-- Runs once, when the Postgres container's data volume is first created.
-- The backend also creates this table on startup if it is missing.
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    predicted_digit INTEGER NOT NULL,
    true_label INTEGER CHECK (true_label BETWEEN 0 AND 9)
);
