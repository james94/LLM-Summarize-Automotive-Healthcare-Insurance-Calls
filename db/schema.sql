create table if not exists public.ah_call_summaries (
    call_id text primary key,
    summary text not null,
    intent_label text not null,
    intent_confidence double precision not null default 0.0,
    key_entities jsonb not null default '{}'::jsonb,
    action_items jsonb not null default '[]'::jsonb,
    status text not null default 'open',
    model_raw jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);
