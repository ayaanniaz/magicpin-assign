# Vera Bot — magicpin AI Challenge Submission

## Approach

Vera is built around a **trigger-routed composer**: instead of one generic prompt, each `trigger.kind` maps to a bespoke prompt frame that knows exactly which context fields to extract and which compulsion levers to activate.

### Architecture

```
POST /v1/context  →  in-memory context store (idempotent by scope + version)
POST /v1/tick     →  trigger iterator → compose() → action list
POST /v1/reply    →  intent detector → auto-reply / opt-out / commit / follow-up
GET  /v1/healthz  →  liveness + context counts
GET  /v1/metadata →  team identity
```

### Composer (`composer.py`)

`compose(category, merchant, trigger, customer?)` does:

1. **Route by `trigger.kind`** — 18+ distinct prompt frames covering: `research_digest`, `regulation_change`, `recall_due`, `perf_dip`, `perf_spike`, `festival_upcoming`, `ipl_match_today`, `review_theme_emerged`, `milestone_reached`, `active_planning_intent`, `winback_eligible`, `renewal_due`, `gbp_unverified`, `supply_alert`, `chronic_refill_due`, `cde_opportunity`, `competitor_opened`, `customer_lapsed_*`, `curious_ask_due`.

2. **Extract grounded facts** — never fabricate. Every number, date, batch number, or citation in the body is pulled from the supplied context fields.

3. **Pick compulsion levers** — chosen per merchant state:
   - **Specificity / verifiability** — concrete numbers from context
   - **Loss aversion** — e.g. "22 of your chronic-Rx customers were dispensed these batches"
   - **Social proof** — peer CTR benchmarks, peer-median comparisons
   - **Effort externalization** — "I'll draft it — just say yes"
   - **Curiosity** — seasonal trends, research findings
   - **Single binary CTA** — one clear commit per message

4. **Voice match** — 5 per-category system prompts (peer_clinical for dentists, coaching for gyms, etc.) + customer-facing overrides.

5. **LLM call** — Gemini 2.5 Flash at `temperature=0` for determinism.

### Conversation engine (`bot.py`)

- **Auto-reply detection**: canned-phrase match → try once → wait 24h → end gracefully
- **Opt-out detection**: explicit "stop messaging" → polite exit + suppress
- **Intent transition**: explicit commit phrases → switch to action mode immediately (no more qualifying)
- **Off-topic handling**: decline politely → redirect to original thread
- **Multi-turn follow-up**: LLM-generated contextual response grounded in conversation history
- **Anti-repetition**: bot never sends the same body twice in the same conversation

## Model choice

**Gemini 2.5 Flash** — best quality-to-latency ratio for this task. All calls at temperature=0 for determinism. Typically responds in 3–8 seconds well within the 30s budget.

## Tradeoffs

| Decision | Rationale |
|---|---|
| 18+ trigger-specific prompts | Generic prompts produce generic messages. Specificity requires knowing what to extract per trigger type. |
| Temperature=0 | Mandatory for determinism. Gemini Flash is still highly capable at this setting. |
| In-memory state | Simpler than Redis for a 60-min test window. Would need persistence for production. |
| No RAG/retrieval | Dataset is small and pre-pushed; direct context injection is faster and avoids retrieval latency. |

## What additional context would help most

1. **Real merchant GBP data** — actual photo counts, post dates, review responses to compute more specific gap metrics
2. **Customer phone numbers** — to differentiate WhatsApp channel for merchant-facing vs customer-facing
3. **Vera's current conversation cadence** — to know when NOT to send (avoid over-messaging the same merchant)

## Local run

```bash
# Copy env
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# Install
pip install -r requirements.txt

# Start
uvicorn bot:app --host 0.0.0.0 --port 8080

# Test
python judge_simulator.py   # after setting LLM_PROVIDER and LLM_API_KEY in judge_simulator.py
```
