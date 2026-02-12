# LLM-Summarize-Automotive-Healthcare-Insurance-Calls

LLM-backed pipeline that reads call-center transcript JSON files (single file or a directory), summarizes each call, classifies intent into a small taxonomy, extracts a few key entities, and upserts results into Supabase.

This repo is intended to be used with the Hugging Face dataset **“92k-real-world-call-center-scripts-english”** (PII-redacted). The current code assumes each transcript JSON has a top-level `text` field.

## What it does

Given a transcript:

- Produces a concise summary
- Chooses one intent label from:
  - `CANCELLATION_REQUEST`, `RESHOP_RATE`, `BILLING_PAYMENT`, `CLAIM_STATUS`, `NEW_QUOTE`, `GENERAL_INQUIRY`, `OTHER`
- Extracts (if present): `policy_number`, `claim_id`, `effective_date`, `amount`
- Generates `action_items` and sets `status` (`open|resolved|follow_up`)
- Writes results to Supabase (via `supabase-py` by default; REST client is included)

## Architecture (main components)

- **Prompt builder**: `build_prompt(transcript, TAXONOMY)`
- **LLM clients**:
  - `HFHubInferenceClient` (huggingface_hub `InferenceClient`)
  - `HFOpenAICompatClient` (OpenAI SDK against HF Router-compatible endpoint)
- **Validation/coercion**:
  - `extract_json_object()` attempts to pull a JSON object from model output
  - `coerce_to_llm_result()` enforces taxonomy + defaults
- **Storage**:
  - `SupabaseStorageClient` (supabase-py upsert)
  - `SupabaseRESTStorageClient` (direct REST upsert with retries)
- **Pipeline**:
  - `CallProcessor.process(event)` -> calls LLM -> builds record -> upserts to Supabase

## Dataset expectations

This code expects each input `.json` transcript file to contain:

- `text` (string): the transcript content

If you want to “zone in” on **automotive and healthcare insurance inbound** calls, you currently do that by selecting/filtering which JSON files you pass in (e.g., by pre-filtering into a directory). The script does not yet implement domain/topic filtering itself.

See dataset details here:
- `/home/james/src/dataset/92k-real-world-call-center-scripts-english/README.md`

## Setup

### 1) Install dependencies

From your project environment:

- `pydantic`
- `python-dotenv`
- `huggingface_hub`
- `openai`
- `supabase`
- `requests`

### 2) Configure environment variables

`src/main.py` loads `.env` from the project root:

- `HF_TOKEN` = Hugging Face token (required for HF inference)
- `HF_MODEL_ID` = model repo id or inference model id (e.g., a chat-capable model)
- `SUPABASE_URL` = `https://<project_ref>.supabase.co`
- `SUPABASE_SECRET_KEY` = service role key (server-side only)
- `SUPABASE_TABLE` = defaults to `ah_call_summaries` if unset

## Usage

### Summarize a single transcript JSON

```bash
python -m src.main --dataset-json /path/to/transcript.json
```

### Summarize a directory (batch)

```bash
python -m src.main --dataset-dir /path/to/json_dir --limit 25
```

Output:
- For a directory run: prints `{ "count": N, "items": [...] }`
- For a single file: prints the full record (summary, intent, entities, etc.)

## Supabase table notes

The code upserts on `call_id` (derived from filename stem). Your table should have a unique constraint on `call_id` to support merge-upserts.

Stored fields include:
- `summary`, `intent_label`, `intent_confidence`, `key_entities`, `action_items`, `status`
- `model_raw` (provider/model metadata + raw response snapshot)

## Operational notes / known gaps

- Model outputs are expected to be JSON; if the model wraps JSON in markdown fences, the parser strips them.
- If parsing fails, the pipeline records an `OTHER` intent and flags `follow_up`.
- Filtering specifically to “automotive” + “healthcare insurance” + “inbound” is not implemented in code yet (selection is currently done by choosing the appropriate input files).

## License / data usage

- Dataset license: see the dataset README (CC BY-NC 4.0; non-commercial).
- Ensure you comply with dataset terms and your model/provider terms.