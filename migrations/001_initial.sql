CREATE TABLE users (
 id uuid PRIMARY KEY, email text UNIQUE NOT NULL, password_hash text NOT NULL,
 role text NOT NULL DEFAULT 'user' CHECK(role IN ('user','admin')), created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE models (
 version text PRIMARY KEY, manifest_sha256 text NOT NULL, threshold double precision NOT NULL,
 evaluation jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE routing_config (id integer PRIMARY KEY CHECK(id=1), active_model text NOT NULL REFERENCES models(version));
CREATE TABLE tickets (
 id uuid PRIMARY KEY, user_id uuid NOT NULL REFERENCES users(id), text text NOT NULL CHECK(length(text) BETWEEN 3 AND 8000),
 language text NOT NULL, idempotency_key text NOT NULL, request_hash text NOT NULL,
 model_version text NOT NULL REFERENCES models(version), status text NOT NULL DEFAULT 'pending'
 CHECK(status IN ('pending','queued','processing','auto_routed','needs_review','reviewed')),
 generation uuid, lease_until timestamptz, attempts integer NOT NULL DEFAULT 0 CHECK(attempts>=0),
 predicted_intent text, department text, confidence double precision CHECK(confidence BETWEEN 0 AND 1),
 threshold double precision, top_choices jsonb, review_reason text,
 final_intent text, final_department text, version integer NOT NULL DEFAULT 1,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(user_id,idempotency_key)
);
CREATE INDEX tickets_due ON tickets(created_at) WHERE status IN ('pending','queued','processing');
CREATE INDEX tickets_review ON tickets(created_at) WHERE status='needs_review';
CREATE TABLE ticket_events (
 id bigserial PRIMARY KEY, ticket_id uuid NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
 type text NOT NULL, details jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE reviews (
 id bigserial PRIMARY KEY, ticket_id uuid NOT NULL REFERENCES tickets(id), reviewer_id uuid NOT NULL REFERENCES users(id),
 ticket_version integer NOT NULL, intent text NOT NULL, note text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), UNIQUE(ticket_id,ticket_version)
);
CREATE TABLE worker_heartbeats (name text PRIMARY KEY, updated_at timestamptz NOT NULL DEFAULT now());
