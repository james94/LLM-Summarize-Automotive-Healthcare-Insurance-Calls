# Steps

# 1. Config/Env

# Moved env vars to .env file

# 2. Taxonomy

TAXONOMY = [
    "CANCELLATION_REQUEST",
    "RESHOP_RATE",
    "BILLING_PAYMENT",
    "CLAIM_STATUS",
    "NEW_QUOTE",
    "GENERAL_INQUIRY",
    "OTHER"
]

# NOTE: you could have a secondary list of taxonomies that the parent
# points to

# 3. Schemas

# pip install pydantic
import os
import json
from pydantic import BaseModel
from pydantic import Field
from typing import Literal, List, Dict, Any

class CallEvent(BaseModel):
    call_id: str
    transcript: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

Status = Literal["open", "resolved", "follow_up"]

class KeyEntities(BaseModel):
    policy_number: str = None
    claim_id: str = None
    effective_date: str = None
    amount: str = None

class LLMResult(BaseModel):
    summary: str
    intent_label: str
    intent_confidence: float = 0.0
    key_entities: KeyEntities = Field(default_factory=KeyEntities)
    action_items: List[str] = Field(default_factory=list)
    status: Status = "open"

# 4. Interfaces
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def summarize_and_classify(self, event: CallEvent) -> tuple[LLMResult, Dict[str, Any]]:
        """Returns tuple(LLMResult, Model Raw)"""

class StorageClient(ABC):
    @abstractmethod
    def upsert_call_summary(self, event: CallEvent, record: Dict[str, Any]) -> None:
        """Sends upsert call summary to update Supabase postgresql table"""


# 5. Prompt Builder

def build_prompt(transcript: str, taxonomy: List[str]) -> str:
    taxonomy_str = ", ".join(taxonomy)

    return f"""
You are a call-center transcript processor for an insurance company.

TASK:
1) Summarize the conversation concisely.
2) Choose the single best intent label from this allowed list: {taxonomy_str}
3) Extract key entities if they are present (policy number, claim id, effective date, amount)
4) Provide action items and a status

OUTPUT FORMAT:
Returns ONLY a valid JSON (no markdown, no backticks). Must match this JSON schema exactly:

{{
  "summary": "string",
  "intent_label": "a label from the allowed list of labels",
  "intent_confidence": 0.0,
  "key_entities": {{
    "policy_number": null,
    "claim_id": null,
    "effective_date": null,
    "amount": null
  }},
  "action_items": ["string"],
  "status": "open|resolved|follow_up"
}}

TRANSCRIPT:
{transcript}
""".strip()

# 6. HF Provider (OpenAI, Hugging Face, REST API)

# NOTE: look up the hugging face url

# pip install openai
# pip install huggingface_hub

from huggingface_hub import InferenceClient

class HFHubInferenceClient(LLMClient):
    def __init__(self, 
        token: str = None,
        model_id: str = None,
        timeout_s: float = 20.0
    ):

        self.token = token or os.environ.get("HF_TOKEN")
        self.model_id = model_id or os.environ.get("HF_MODEL_ID")
        self.timeout_s = timeout_s

        self.client = InferenceClient(api_key=self.token, timeout=self.timeout_s)

    def summarize_and_classify(self, event: CallEvent) -> tuple[LLMResult, Dict[str, Any]]:
        prompt = build_prompt(event.transcript, TAXONOMY)

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )

        generated_text = ""

        try:
            generated_text = completion.choices[0].message.content or ""
        except Exception:
            generated_text = str(completion)

        llm_result = coerce_to_llm_result(generated_text)

        model_raw: Dict[str, Any] = {
            "provider": "huggingface_hub_inference_client",
            "model_id": self.model_id,
            "raw_response": getattr(completion, "model_dump", lambda: str(completion))(),
            "generated_text": generated_text,
        }

        return llm_result, model_raw

from openai import OpenAI

class HFOpenAICompatClient(LLMClient):
    def __init__(self,
        token: str = None,
        model_id: str = None,
        base_url: str = "https://router.huggingface.co/v1",
        timeout_s: float = 20.0
    ):
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_KEY")
        self.model_id = model_id or os.environ.get("HF_MODEL_ID")
        self.timeout_s = timeout_s
        self.base_url = base_url

        self.client = OpenAI(base_url=self.base_url, api_key=self.token)

    def summarize_and_classify(self, event: CallEvent) -> tuple[LLMResult, Dict[str, Any]]:
        prompt = build_prompt(event.transcript, TAXONOMY)

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
            timeout=self.timeout_s,
        )

        generated_text = ""

        try:
            generated_text = completion.choices[0].message.content or ""
        except Exception:
            generated_text = str(completion)

        llm_result = coerce_to_llm_result(generated_text)

        model_raw: Dict[str, Any] = {
            "provider": "openai_sdk_to_hf_router",
            "model_id": self.model_id,
            "raw_response": getattr(completion, "model_dump", lambda: str(completion))(),
            "generated_text": generated_text,
        }

        return llm_result, model_raw

# 7. Validation


import re

# Extract JSON Object
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # clean
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    # check for match for JSON {...} from LLM model's output
    m = _JSON_OBJECT_RE.search(cleaned)

    if not m:
        raise RuntimeError("No match for JSON {...} found in LLM model's output")

    return json.loads(m.group(0))

# Coerce to LLM Result

def coerce_to_llm_result(model_text: str) -> LLMResult:
    try:
        obj = extract_json_object(model_text)
    except Exception:
        return LLMResult(
            summary="Unable to parse the transcript; manual review required.",
            intent_label="OTHER",
            intent_confidence=0.0,
            action_items=["Review transcript and correct intent/summary."],
            status="follow_up",
        )

    # Enforce label
    label = obj.get("intent_label")
    if label not in TAXONOMY:
        label = "OTHER"

    # intent_confidence: coerce to float safely
    raw_conf = obj.get("intent_confidence", 0.0)
    try:
        conf = float(raw_conf)
    except Exception:
        conf = 0.0

    # Extract Key Entities
    ke = obj.get("key_entities")

    if not isinstance(ke, dict):
        ke = {}

    llm_result = LLMResult(
        summary=str(obj.get("summary") or "No summary provided").strip(),
        intent_label=label,
        intent_confidence=conf,
        key_entities=KeyEntities(
            policy_number=ke.get("policy_number"),
            claim_id=ke.get("claim_id"),
            effective_date=ke.get("effective_date"),
            amount=ke.get("amount"),
        ),
        action_items=[str(x) for x in (obj.get("action_items") or []) if str(x).strip()],
        status=obj.get("status") if obj.get("status") in ("open", "resolved", "follow_up") else "open",
    )

    return llm_result


# 8. Storage (Supabase: REST API, supabase-py)

# Create Supabase REST Storage

# NOTE: what is the appropriate url for upsert call to supabase via REST API Call?

# pip install requests

# NOTE: https://<project_ref>.supabase.co/rest/v1/<table_name>?on_conflict=<column_name(s)>

import time
import requests

# pip install supabase

import supabase

class SupabaseStorageClient(StorageClient):
    def __init__(self,
        url: str = None,
        table: str = None,
        key: str = None
    ):
        self.url = (url or os.environ.get("SUPABASE_URL")).rstrip("/")
        self.table = table or os.environ.get("SUPABASE_TABLE") or "ah_call_summaries"
        self.key = key or os.environ.get("SUPABASE_SECRET_KEY")
        
        self.client = supabase.create_client(self.url, self.key)

    def upsert_call_summary(self, event: CallEvent, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload["call_id"] = event.call_id

        resp = self.client.table(self.table).upsert(payload).execute()

        print(f"upsert response from supabase client: {resp}")

# TODO: we should check why Supabase REST API upsert call was failing

class SupabaseRESTStorageClient(StorageClient):
    def __init__(self,
        url: str = None,
        table: str = None,
        key: str = None,
        timeout_s: float = 20.0,
        max_retries: int = 2,
    ):
        self.url = (url or os.environ.get("SUPABASE_URL")).rstrip("/")
        self.table = table or os.environ.get("SUPABASE_TABLE") or "ah_call_summaries"
        self.key = key or os.environ.get("SUPABASE_SECRET_KEY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def _upsert_url(self):
        return f"{self.url}/rest/v1/{self.table}?on_conflict=call_id"

    def upsert_call_summary(self, event: CallEvent, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload["call_id"] = event.call_id

        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates, return=minimal"
        }

        last_err: Exception = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self._upsert_url(),
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )

                # Optional: check transient failure
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Encountered transient failure on Supabase upsert call {resp.status_code}: {resp.text[:2000]}")

                # Check for hard failure
                if resp.status_code >= 400:
                    raise RuntimeError(f"Encountered hard failure on Supabase upsert call {resp.status_code}: {resp.text[:4000]}")

                # We dont encounter those failures above, success on REST API upsert call summary
                return

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (1 + attempt))

        print(f"Potentially ran into issue sending REST API upsert call summary to Supabase: {last_err!r}")

# Create the table: ah_call_summaries

# NOTE: wrote SQL create table code in schema.sql script; Added to Supabase SQL editor

# 9. Pipeline (Call Processor)

class CallProcessor:
    def __init__(self, llm: LLMClient, storage: StorageClient):
        self.llm = llm
        self.storage = storage

    def process(self, event: CallEvent) -> Dict[str, Any]:
        llm_result, model_raw = self.llm.summarize_and_classify(event)

        record: Dict[str, Any] = {
            "summary": llm_result.summary,
            "intent_label": llm_result.intent_label,
            "intent_confidence": llm_result.intent_confidence,
            "key_entities": llm_result.key_entities.model_dump(),
            "action_items": llm_result.action_items,
            "status": llm_result.status,
            "model_raw": model_raw
        }

        self.storage.upsert_call_summary(event, record)

        return record

# 10. Entrypoint

# pip install dotenv
import argparse
from pathlib import Path
from dotenv import load_dotenv

def _event_from_dataset_json_path(p: Path) -> CallEvent:
    obj = json.loads(p.read_text(encoding="utf-8"))
    transcript = str(obj.get("text")).strip()

    call_id = p.stem

    # NOTE: we could also set asr_confidence, audio duration, redacted pii policies
    metadata = {}

    event = CallEvent(call_id=call_id, transcript=transcript, metadata=metadata)

    return event

def main():
    # Part 1) load env vars
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

    # NOTE: for future improvement, we could add checks for env vars

    # Part 2) create argument parser and parse args

    ap = argparse.ArgumentParser()

    mx = ap.add_mutually_exclusive_group(required=True)

    # NOTE: if later we needed to account for .txt transcript file, we could add an argument for it
    mx.add_argument("--dataset-json", help="Path to single .json transcript file (expect top-level 'text')")
    mx.add_argument("--dataset-dir", help="Path to directory of .json transcript files")

    ap.add_argument("--limit", type=int, default=5, help="Number of .json transcript to parse in dataset folder")

    args = ap.parse_args()

    # Part 3) create CallProcessor
    # processor = CallProcessor(llm=HFOpenAICompatClient(), storage=SupabaseRESTStorage())
    processor = CallProcessor(llm=HFHubInferenceClient(), storage=SupabaseStorageClient())

    # Part 4) Parse the .json transcript files in the dataset directory
    if args.dataset_dir:
        d = Path(args.dataset_dir)

        # check if directory exists or is a directory
        if not d.exists() or not d.is_dir():
            raise RuntimeError("--dataset-dir must exist and be a directory")

        # extract json filenames
        json_files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json"]

        # check limit >= 1
        if args.limit < 1:
            raise RuntimeError("--limit must be >= 1")

        # process and summarize .json transcripts into processed list of dict
        processed: List[dict] = []

        for p in json_files[: args.limit]:
            event = _event_from_dataset_json_path(p)
            record = processor.process(event)

            processed.append({"path": str(p), "call_id": event.call_id, "intent_label": record["intent_label"]})

        # print json dumps count processed, items processed
        print(json.dumps({"count": len(processed), "items": processed}, indent=2))

        return
    
    if args.dataset_json:
        p = Path(args.dataset_json)

        event = _event_from_dataset_json_path(p)

        record = processor.process(event)

        # print json dumps record
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
