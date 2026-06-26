"""
bot.py — Vera Bot HTTP server
FastAPI implementation of all 5 required endpoints + conversation state.
"""

from __future__ import annotations
import os, time, uuid, json, re
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY from .env if present

from composer import compose, _is_auto_reply

app = FastAPI(title="Vera Bot", version="1.0.0")
START = time.time()

# ── In-memory state ─────────────────────────────────────────────────────────
# (scope, context_id) → {version: int, payload: dict}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id → {merchant_id, customer_id, turns: list, suppressed: bool, wait_until: float}
conversations: dict[str, dict] = {}

# Set of suppression keys already sent (dedup)
sent_suppression_keys: set[str] = set()

# Auto-reply tracking: conversation_id → count of consecutive auto-replies
auto_reply_counts: dict[str, int] = {}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _counts() -> dict:
    c = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _ in contexts:
        if scope in c:
            c[scope] += 1
    return c


def _get_payload(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None


def _conv_bodies(conv_id: str) -> list[str]:
    """All bot-sent bodies in a conversation (for anti-repetition)."""
    conv = conversations.get(conv_id, {})
    return [t["body"] for t in conv.get("turns", []) if t.get("from") == "bot"]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": _counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Intelligence",
        "team_members": ["Ayaan Bandey"],
        "model": "gemini-3-flash-preview",
        "approach": (
            "Trigger-routed composer: each trigger.kind maps to a bespoke prompt frame "
            "that grounds specificity in real context fields. "
            "Compulsion levers (loss aversion, social proof, effort externalisation) "
            "are selected per merchant state. Temperature=0 for determinism. "
            "Auto-reply detection, intent-transition routing, and graceful exit included."
        ),
        "contact_email": "ayaanniaz777@gmail.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(
            status_code=400,
            content={
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"Unknown scope: {body.scope}",
            },
        )
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": cur["version"],
            },
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    now_str = body.now

    for trg_id in body.available_triggers:
        # Respect action cap
        if len(actions) >= 20:
            break

        trg_payload = _get_payload("trigger", trg_id)
        if not trg_payload:
            continue

        # Check expiry
        expires_at = trg_payload.get("expires_at", "")
        if expires_at:
            try:
                exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if exp < datetime.now(timezone.utc):
                    continue  # expired
            except Exception:
                pass

        # Suppression check
        sup_key = trg_payload.get("suppression_key", "")
        if sup_key and sup_key in sent_suppression_keys:
            continue

        merchant_id = trg_payload.get("merchant_id")
        customer_id = trg_payload.get("customer_id")

        merchant = _get_payload("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        cat_slug = merchant.get("category_slug", "")
        category = _get_payload("category", cat_slug)
        if not category:
            continue

        customer = _get_payload("customer", customer_id) if customer_id else None

        # Generate conversation_id
        conv_id = f"conv_{merchant_id}_{trg_id}"

        # Skip if conversation is suppressed
        conv = conversations.get(conv_id, {})
        if conv.get("suppressed"):
            continue

        # Skip if waiting
        wait_until = conv.get("wait_until", 0)
        if wait_until > time.time():
            continue

        try:
            result = compose(category, merchant, trg_payload, customer)
        except Exception as e:
            continue

        body_text = result.get("body", "")
        if not body_text:
            continue

        # Anti-repetition: skip if same body sent in this conversation
        prior_bodies = _conv_bodies(conv_id)
        if body_text in prior_bodies:
            continue

        # Record
        conversations.setdefault(
            conv_id,
            {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "turns": [],
                "suppressed": False,
                "wait_until": 0,
            },
        )
        conversations[conv_id]["turns"].append(
            {"from": "bot", "body": body_text, "ts": now_str}
        )
        if sup_key:
            sent_suppression_keys.add(sup_key)

        scope = trg_payload.get("scope", "merchant")
        send_as = result.get("send_as", "vera")
        cta = result.get("cta", "open_ended")
        suppression_key = result.get("suppression_key", sup_key)
        rationale = result.get("rationale", "")

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg_payload.get('kind','generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("owner_first_name", ""),
                body_text[:80],
                "",
            ],
            "body": body_text,
            "cta": cta,
            "suppression_key": suppression_key,
            "rationale": rationale,
        }
        actions.append({"action": "send_message", "payload": action})

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    message = body.message
    merchant_id = body.merchant_id
    customer_id = body.customer_id

    # Ensure conversation exists
    conv = conversations.setdefault(
        conv_id,
        {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "turns": [],
            "suppressed": False,
            "wait_until": 0,
        },
    )

    # Record incoming turn
    conv["turns"].append(
        {"from": body.from_role, "body": message, "ts": body.received_at}
    )

    # ── Auto-reply detection ─────────────────────────────────────────────
    if _is_auto_reply(message):
        count = auto_reply_counts.get(conv_id, 0) + 1
        auto_reply_counts[conv_id] = count

        if count == 1:
            # First auto-reply: send one explicit nudge
            nudge = (
                "Looks like an auto-reply 😊 When the owner sees this, "
                "just reply 'Yes' to continue."
            )
            conv["turns"].append({"from": "bot", "body": nudge})
            return {
                "action": "send",
                "body": nudge,
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply (canned phrasing). Sending one explicit nudge for the owner.",
            }
        elif count == 2:
            # Second auto-reply: wait 24h
            conv["wait_until"] = time.time() + 86400
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Auto-reply for 2nd time. Owner not at phone. Backing off 24h.",
            }
        else:
            # 3+ auto-replies: end
            conv["suppressed"] = True
            return {
                "action": "end",
                "rationale": "Auto-reply detected 3+ consecutive times. No real engagement. Closing conversation.",
            }
    else:
        # Reset auto-reply counter on real reply
        auto_reply_counts[conv_id] = 0

    # ── Hard opt-out detection ───────────────────────────────────────────
    msg_lower = message.lower()
    OPT_OUT_SIGNALS = [
        "not interested",
        "stop messaging",
        "stop sending",
        "unsubscribe",
        "don't message",
        "don't contact",
        "remove me",
        "opt out",
        "why are you bothering",
        "this is spam",
        "useless",
    ]
    if any(s in msg_lower for s in OPT_OUT_SIGNALS):
        conv["suppressed"] = True
        return {
            "action": "send",
            "body": "Apologies — I won't message again. If anything changes, you can always restart with 'Hi Vera'. 🙏",
            "cta": "none",
            "rationale": "Merchant expressed disinterest/hostility. Sending polite exit message and closing.",
        }

    # ── Intent transition: explicit commit ───────────────────────────────
    COMMIT_SIGNALS = [
        "let's do it",
        "lets do it",
        "ok go ahead",
        "yes go ahead",
        "proceed",
        "confirm",
        "sounds good, go",
        "do it",
        "send it",
        "yes please",
        "yes do it",
        "perfect do it",
    ]
    if any(s in msg_lower for s in COMMIT_SIGNALS) or msg_lower.strip() in (
        "yes",
        "1",
        "confirm",
        "go",
    ):
        # Action mode: deliver the next concrete step
        merchant = _get_payload("merchant", merchant_id or conv.get("merchant_id", ""))
        cat_slug = (merchant or {}).get("category_slug", "")
        category = _get_payload("category", cat_slug)
        active_offers = [
            o["title"]
            for o in (merchant or {}).get("offers", [])
            if o.get("status") == "active"
        ]
        mname = (merchant or {}).get("identity", {}).get("name", "your business")
        owner = (merchant or {}).get("identity", {}).get("owner_first_name", "")

        # Build a concrete follow-through body
        if active_offers:
            next_body = (
                f"On it, {owner}! Drafting your campaign materials now — 60 seconds. "
                f"I'll set up: (1) a GBP post featuring {active_offers[0]}, "
                f"(2) a WhatsApp broadcast template for your customer list. "
                f"Reply CONFIRM to proceed, or tell me what to adjust."
            )
        else:
            next_body = (
                f"On it, {owner}! Drafting your next steps now. "
                f"Reply CONFIRM to proceed, or tell me what to tweak."
            )

        conv["turns"].append({"from": "bot", "body": next_body})
        return {
            "action": "send",
            "body": next_body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed with explicit intent signal. Switching from qualifying to action-execution mode.",
        }

    # ── Off-topic / out-of-scope detection ──────────────────────────────
    OUT_OF_SCOPE = [
        "gst filing",
        "income tax",
        "itr",
        "loan",
        "insurance claim",
        "legal advice",
        "property",
        "astrology",
    ]
    if any(s in msg_lower for s in OUT_OF_SCOPE):
        # Redirect to the original topic
        last_bot_body = next(
            (
                t["body"]
                for t in reversed(conv.get("turns", []))
                if t.get("from") == "bot"
            ),
            "the topic I mentioned",
        )
        redirect = (
            "That's outside what I can help with directly — "
            "you'd want a specialist for that. "
            "Coming back to where we left off — want me to proceed with the draft?"
        )
        conv["turns"].append({"from": "bot", "body": redirect})
        return {
            "action": "send",
            "body": redirect,
            "cta": "open_ended",
            "rationale": "Out-of-scope request politely declined. Redirected to original thread without losing context.",
        }

    # ── General reply: compose a context-aware follow-up ────────────────
    merchant = _get_payload("merchant", merchant_id or conv.get("merchant_id", ""))
    cat_slug = (merchant or {}).get("category_slug", "")
    category = _get_payload("category", cat_slug)
    customer_ctx = (
        _get_payload("customer", customer_id or conv.get("customer_id", ""))
        if customer_id
        else None
    )

    if merchant and category:
        # Build a multi-turn follow-up using the composer's LLM
        from composer import (
            _llm,
            VOICE_SYSTEM,
            _owner_first,
            _merchant_name,
            _active_offers,
            _language_note,
        )

        slug = category.get("slug", "default")
        system = VOICE_SYSTEM.get(slug, VOICE_SYSTEM["default"])
        owner_name = _owner_first(merchant)
        mname = _merchant_name(merchant)
        active_offers = _active_offers(merchant)
        lang_note = _language_note(merchant)

        # Build conversation context for the LLM
        conv_str = "\n".join(
            f"{'BOT' if t.get('from')=='bot' else 'MERCHANT'}: {t['body']}"
            for t in conv.get("turns", [])[-6:]  # last 6 turns
        )
        prior_bodies = _conv_bodies(conv_id)

        prompt = f"""Continue this WhatsApp conversation naturally.

MERCHANT: {mname}, {slug}
ACTIVE OFFERS: {active_offers}

CONVERSATION SO FAR:
{conv_str}

LATEST MERCHANT MESSAGE: "{message}"

RULES:
1. Respond to what the merchant just said — directly and helpfully.
2. If they're asking a question, answer it with a concrete fact or action.
3. If they're giving feedback, acknowledge + advance the conversation.
4. Keep it SHORT (≤70 words). One CTA at most.
5. {lang_note}
6. NO URLs. No preamble. Don't repeat what was already said.

Output ONLY the response body. No quotes. No markdown."""

        try:
            follow_up = _llm(prompt, system).strip().strip('"').strip("'").strip()
        except Exception:
            follow_up = (
                f"Got it, {owner_name}! Want me to draft that for you right now? "
                f"Just say yes and I'll have it ready in 60 seconds."
            )

        # Anti-repetition
        if follow_up in prior_bodies:
            follow_up = f"Here's what I can do next — want me to proceed? Reply YES."

        conv["turns"].append({"from": "bot", "body": follow_up})
        return {
            "action": "send",
            "body": follow_up,
            "cta": "open_ended",
            "rationale": f"Multi-turn follow-up to merchant reply. Turn {body.turn_number}.",
        }

    # Fallback
    fallback = (
        "Got it! Want me to put together a quick draft for you? "
        "Just say yes and I'll have it ready."
    )
    conv["turns"].append({"from": "bot", "body": fallback})
    return {
        "action": "send",
        "body": fallback,
        "cta": "binary_yes_no",
        "rationale": "Fallback: no merchant context available for follow-up.",
    }


# Optional teardown (magicpin harness may send this)
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    auto_reply_counts.clear()
    return {"wiped": True}
