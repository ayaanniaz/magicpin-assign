"""
composer.py — Vera's deterministic message engine.

compose(category, merchant, trigger, customer?) → ComposedMessage

Strategy:
  1. Route by trigger.kind to get the right frame (research / recall / dip / festival / …)
  2. Extract grounded facts from all 4 contexts (never invent)
  3. Pick ≤2 compulsion levers matching the merchant's state
  4. Build a tight, voice-matched body with one clear CTA
  5. Return full ComposedMessage dict
"""

from __future__ import annotations
import json, os, re
from typing import Optional
from datetime import datetime, timezone

# ── LLM client (Google Gemini, temperature 0 for determinism) ──────────────
from google import genai
from google.genai import types as genai_types

_MODEL_NAME = "gemini-3-flash-preview"
_MODEL_FALLBACK = "gemini-2.0-flash"
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=key)
    return _client


def _llm(prompt: str, system: str) -> str:
    client = _get_client()
    import time as _time

    _SAFETY_OFF = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
        ),
    ]

    def _call(model: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                max_output_tokens=700,
                safety_settings=_SAFETY_OFF,
            ),
        )
        text = ""
        try:
            text = resp.text.strip()
        except Exception:
            pass
        if not text and resp.candidates:
            for part in resp.candidates[0].content.parts or []:
                if hasattr(part, "text") and part.text:
                    text = part.text.strip()
                    break
        return text

    # Transient error keywords that warrant a retry
    _TRANSIENT = (
        "503",
        "429",
        "unavailable",
        "resource has been exhausted",
        "quota",
        "overloaded",
    )

    last_err = None
    retry_delays = [5, 10, 20, 40, 60]  # seconds between retries (5 attempts per model)

    for model in (_MODEL_NAME, _MODEL_FALLBACK):
        for attempt in range(5):
            try:
                result = _call(model)
                if attempt > 0:
                    print(f"[Vera] {model} succeeded on attempt {attempt + 1}/5")
                return result
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_transient = any(t in err_str for t in _TRANSIENT)
                wait = retry_delays[attempt]  # 5s, 10s, 20s, 40s, 60s
                print(
                    f"[Vera] {model} attempt {attempt + 1}/5 failed"
                    f"{' [transient]' if is_transient else ' [hard]'}: {e}"
                )
                if is_transient and attempt < 4:
                    print(f"[Vera] Waiting {wait}s before retry …")
                    _time.sleep(wait)
                    continue
                # Hard error or final attempt → move to fallback model
                break
        else:
            # All 5 attempts exhausted on this model → try fallback
            print(f"[Vera] All 5 attempts exhausted on {model}, trying fallback …")

    raise RuntimeError(
        f"LLM call failed after 5 retries on both {_MODEL_NAME} and {_MODEL_FALLBACK}. "
        f"Last error: {last_err}"
    ) from last_err


# ── Voice profiles per category ────────────────────────────────────────────
# Shared anti-pattern rules appended to every system prompt
_ANTI_PATTERNS = (
    " STRICT RULES: "
    "(1) NEVER open with 'Hope you're doing well', 'I hope this finds you', 'Just checking in', or ANY pleasantry/preamble. "
    "(2) NEVER re-introduce yourself after the first message. "
    "(3) NEVER use 'guaranteed', 'miracle', 'best in city', or fake superlatives. "
    "(4) DO NOT add URLs. "
    "(5) Keep to ≤100 words unless explicitly asked for more. "
    "(6) Output ONLY the WhatsApp message body — no quotes, no markdown, no surrounding text."
)

VOICE_SYSTEM = {
    "dentists": (
        "You are Vera, magicpin's merchant-AI assistant. You are writing to a dentist. "
        "Tone: peer-collegial, clinical vocabulary welcome (fluoride varnish, caries, OPG, "
        "periodontal, aligner, zirconia, RCT, IOPA). Register: respectful but not formal — "
        "like a smart colleague. Taboo words: 'guaranteed', '100% safe', 'miracle', "
        "'best in city'. Use Dr. <FirstName> salutation. Hindi-English code-mix is fine "
        "if merchant language includes 'hi'. Never fabricate data." + _ANTI_PATTERNS
    ),
    "salons": (
        "You are Vera, magicpin's merchant-AI assistant. You are writing to a salon owner. "
        "Tone: warm, friendly, practical. Use first name. Reference specific services, "
        "stylist names if known, and seasonal demand signals. Avoid clinical or medical language. "
        "Hindi-English mix is fine where appropriate. One clear CTA per message. Never fabricate."
        + _ANTI_PATTERNS
    ),
    "restaurants": (
        "You are Vera, magicpin's merchant-AI assistant. You are writing to a restaurant owner. "
        "Tone: operator-to-operator — fellow business person. Use terms like 'covers', 'AOV', "
        "'footfall', 'delivery share'. Reference their actual dish names, locality, and numbers. "
        "Contrarian advice (backed by data) is valued. Never fabricate."
        + _ANTI_PATTERNS
    ),
    "gyms": (
        "You are Vera, magicpin's merchant-AI assistant. You are writing to a gym owner. "
        "Tone: coaching, motivational, energetic but grounded. Reference member count, "
        "churn rate, trial-to-paid conversion. Use fitness vocabulary (HIIT, strength, "
        "body composition, membership renewal). Warm, not pushy. Never fabricate."
        + _ANTI_PATTERNS
    ),
    "pharmacies": (
        "You are Vera, magicpin's merchant-AI assistant. You are writing to a pharmacy owner. "
        "Tone: trustworthy, precise, compliance-first. Reference batch numbers, molecule names, "
        "regulatory circulars accurately. No hype. Seasonal health trends are fine. "
        "Never fabricate molecule names, recall details, or regulatory data."
        + _ANTI_PATTERNS
    ),
    "default": (
        "You are Vera, magicpin's merchant-AI assistant. Tone: helpful, specific, concise. "
        "Reference only data provided in the context. One CTA per message. Never fabricate."
        + _ANTI_PATTERNS
    ),
}

# Customer-facing overrides
_CUSTOMER_ANTI_PATTERNS = (
    " STRICT RULES: "
    "(1) NEVER use 'Hope you're doing well' or any pleasantry/preamble. "
    "(2) NEVER make medical claims or use 'guaranteed'. "
    "(3) DO NOT add URLs. "
    "(4) Output ONLY the WhatsApp message body — no quotes, no markdown."
)

CUSTOMER_VOICE_SYSTEM = {
    "dentists": (
        "You are drafting a message FROM Dr. {merchant_name}'s clinic TO a patient named {customer_name}. "
        "Tone: warm-clinical. No medical claims. Address the patient by name. "
        "Use Hindi-English mix if patient language pref says 'hi-en'. "
        "Do NOT use 'cure', 'guaranteed', or alarm language. "
        "Keep it caring and action-oriented." + _CUSTOMER_ANTI_PATTERNS
    ),
    "salons": (
        "You are drafting a message FROM {merchant_name} TO a customer named {customer_name}. "
        "Tone: warm, friendly, personal. Reference their service history if available. "
        "Use their language preference. One clear booking CTA."
        + _CUSTOMER_ANTI_PATTERNS
    ),
    "restaurants": (
        "You are drafting a message FROM {merchant_name} TO a customer named {customer_name}. "
        "Tone: casual, friendly. Reference their favourite dish if known. "
        "Keep it short and action-oriented." + _CUSTOMER_ANTI_PATTERNS
    ),
    "gyms": (
        "You are drafting a message FROM {merchant_name} TO a member named {customer_name}. "
        "Tone: warm, encouraging, no shame or guilt. Reference their training focus if known. "
        "No-commitment CTA preferred." + _CUSTOMER_ANTI_PATTERNS
    ),
    "pharmacies": (
        "You are drafting a message FROM {merchant_name} TO a customer. "
        "Tone: trustworthy, respectful. For seniors, use formal address (Sharma ji, Namaste). "
        "Precise with molecule names and dates. Include delivery / call option."
        + _CUSTOMER_ANTI_PATTERNS
    ),
    "default": (
        "You are drafting a message FROM {merchant_name} TO {customer_name}. "
        "Tone: friendly, helpful. One CTA. Language-pref aware."
        + _CUSTOMER_ANTI_PATTERNS
    ),
}


# ── Fact extraction helpers ────────────────────────────────────────────────


def _safe(d: dict, *keys, default=""):
    """Safe nested dict getter."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d != {} else default


def _active_offers(merchant: dict) -> list[str]:
    return [
        o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"
    ]


def _merchant_name(merchant: dict) -> str:
    return _safe(merchant, "identity", "name") or "the clinic"


def _owner_first(merchant: dict) -> str:
    return _safe(merchant, "identity", "owner_first_name") or _merchant_name(merchant)


def _locality(merchant: dict) -> str:
    loc = _safe(merchant, "identity", "locality")
    city = _safe(merchant, "identity", "city")
    return f"{loc}, {city}" if loc and city else loc or city or ""


def _ctr(merchant: dict) -> float:
    return float(_safe(merchant, "performance", "ctr") or 0)


def _peer_ctr(category: dict) -> float:
    return float(_safe(category, "peer_stats", "avg_ctr") or 0)


def _digest_item(category: dict, item_id: str) -> Optional[dict]:
    for d in category.get("digest", []):
        if d.get("id") == item_id:
            return d
    return None


def _language_note(merchant: dict) -> str:
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return "Use natural Hindi-English code-mix (like: 'Aapke liye', 'kya lagta hai', 'bilkul')."
    if "ta" in langs:
        return "You may use a Tamil greeting (Vanakkam) if natural, otherwise English is fine."
    if "te" in langs:
        return "You may use a Telugu greeting if natural, otherwise English is fine."
    return "English only — merchant language is English."


# ── Trigger-kind → composer router ─────────────────────────────────────────


def _build_prompt_for_trigger(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) based on trigger kind."""

    kind = trigger.get("kind", "")
    slug = category.get("slug", "default")
    payload = trigger.get("payload", {})
    owner = _owner_first(merchant)
    mname = _merchant_name(merchant)
    locality = _locality(merchant)
    active_offers = _active_offers(merchant)
    lang_note = _language_note(merchant)
    signals = merchant.get("signals", [])
    perf = merchant.get("performance", {})
    cust_agg = merchant.get("customer_aggregate", {})
    conv_hist = merchant.get("conversation_history", [])
    peer_ctr = _peer_ctr(category)
    my_ctr = _ctr(merchant)
    ctr_vs_peer = (
        "above peer"
        if my_ctr >= peer_ctr
        else f"below peer ({peer_ctr:.1%} peer vs {my_ctr:.1%} yours)"
    )
    review_themes = merchant.get("review_themes", [])

    # Choose system prompt (merchant-facing vs customer-facing)
    if customer:
        cname = _safe(customer, "identity", "name") or "the customer"
        tmpl = CUSTOMER_VOICE_SYSTEM.get(slug, CUSTOMER_VOICE_SYSTEM["default"])
        system = tmpl.format(merchant_name=mname, customer_name=cname)
    else:
        system = VOICE_SYSTEM.get(slug, VOICE_SYSTEM["default"])

    # ── Per-kind prompt factories ──────────────────────────────────────────

    if kind == "research_digest":
        item_id = payload.get("top_item_id", "")
        item = _digest_item(category, item_id) or {}
        title = item.get("title", "A new research finding")
        source = item.get("source", "")
        trial_n = item.get("trial_n", "")
        patient_seg = item.get("patient_segment", "")
        summary = item.get("summary", "")
        actionable = item.get("actionable", "")
        relevant_cohort = ""
        if patient_seg == "high_risk_adults" and cust_agg.get("high_risk_adult_count"):
            relevant_cohort = f"Merchant has {cust_agg['high_risk_adult_count']} high-risk adult patients."

        prompt = f"""Write a SHORT (≤90 words) WhatsApp message from Vera to {owner} about a new research digest item.

MERCHANT FACTS:
- Name: {mname}, {locality}
- CTR: {my_ctr:.1%} ({ctr_vs_peer})
- Signals: {signals}
- {relevant_cohort}

RESEARCH ITEM:
- Title: {title}
- Source: {source}
- Trial N: {trial_n}
- Patient segment: {patient_seg}
- Summary: {summary}
- Actionable: {actionable}

RULES:
1. Open with: "{owner}," (use Dr. prefix for dentists)
2. State the finding with concrete numbers (trial_n, %, source page)
3. Connect it to their specific patient roster or signals
4. Offer one low-effort follow-on (draft content, pull abstract, etc.)
5. End with one open-ended CTA
6. {lang_note}
7. NO URLs. No preamble ("Hope you're well…"). No repetition from conv_history.
8. Append source at end if credible (e.g. — JIDA Oct 2026 p.14)

Output ONLY the WhatsApp message body. No quotes. No markdown."""

    elif kind == "regulation_change":
        item_id = payload.get("top_item_id", "")
        item = _digest_item(category, item_id) or {}
        title = item.get("title", "A regulatory change")
        source = item.get("source", "")
        summary = item.get("summary", "")
        actionable = item.get("actionable", "")
        deadline = payload.get("deadline_iso", "")
        if deadline:
            try:
                dl = datetime.fromisoformat(deadline.replace("Z", ""))
                days_left = (dl - datetime.now()).days
                deadline_str = (
                    f"Deadline: {dl.strftime('%d %b %Y')} ({days_left} days from now)"
                )
            except Exception:
                deadline_str = f"Deadline: {deadline}"
        else:
            deadline_str = ""

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} about a compliance/regulation change.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Active offers: {active_offers}
- Signals: {signals}

REGULATION:
- Title: {title}
- Source: {source}
- Summary: {summary}
- Action required: {actionable}
- {deadline_str}

RULES:
1. Open with owner first name. State urgency but not panic.
2. Give the key change in one sentence with numbers.
3. State clear action: what they must do + by when.
4. Offer to help draft/audit.
5. {lang_note}
6. NO URLs. Append source citation at end.
7. End with binary YES/STOP CTA.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "recall_due":
        cname = _safe(customer, "identity", "name") or "the patient"
        lang_pref = _safe(customer, "identity", "language_pref") or "english"
        last_visit = _safe(customer, "relationship", "last_visit") or ""
        slots = payload.get("available_slots", [])
        service_due = payload.get("service_due", "6_month_cleaning").replace("_", " ")
        due_date = payload.get("due_date", "")
        slot_labels = [s.get("label", "") for s in slots[:2]]
        slot_str = (
            " ya ".join(slot_labels)
            if lang_pref == "hi-en mix"
            else " or ".join(slot_labels)
        )
        active_offer_str = active_offers[0] if active_offers else ""

        # Compute months since last visit
        months_since = ""
        if last_visit:
            try:
                lv = datetime.fromisoformat(last_visit)
                delta = (datetime.now() - lv).days // 30
                months_since = f"{delta} months"
            except Exception:
                months_since = ""

        prompt = f"""Write a SHORT (≤100 words) WhatsApp message FROM {mname} TO patient {cname} for a recall reminder. Send_as: merchant_on_behalf.

PATIENT FACTS:
- Name: {cname}
- Language preference: {lang_pref}
- Last visit: {last_visit} ({months_since} ago)
- Service due: {service_due}
- Slot preferences: {_safe(customer, 'preferences', 'preferred_slots')}

AVAILABLE SLOTS: {slot_str}
ACTIVE OFFER: {active_offer_str}

RULES:
1. Address patient by first name. Name the merchant clinic.
2. State how long since last visit and what is due.
3. Offer 2 specific slots. 
4. Mention offer/price if available.
5. End with: Reply 1 for first slot, 2 for second slot, or suggest a time.
6. {'Use Hindi-English mix naturally (e.g. Apke liye slots ready hain)' if 'hi' in lang_pref else 'English only.'}
7. NO URLs. Warm-clinical tone. No medical claims.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "perf_dip":
        metric = payload.get("metric", "calls")
        delta_pct = payload.get("delta_pct", 0)
        window = payload.get("window", "7d")
        baseline = payload.get("vs_baseline", "")
        is_seasonal = payload.get("is_expected_seasonal", False)
        season_note = payload.get("season_note", "")

        prompt = f"""Write a SHORT (≤85 words) WhatsApp message from Vera to {owner} about a performance dip.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Metric: {metric}
- Drop: {abs(delta_pct)*100:.0f}% over {window} (vs baseline {baseline})
- Seasonal: {"Yes — " + season_note if is_seasonal else "No — unexplained"}
- CTR vs peer: {ctr_vs_peer}
- Active offers: {active_offers}
- Signals: {signals}
- Review themes: {review_themes}

RULES:
1. Open with owner name. State the exact drop % and metric.
2. If seasonal, reframe it as expected + give the smart action (save spend, focus retention).
3. If not seasonal, diagnose and offer a fix tied to their specific offers/profile signals.
4. Give one concrete proposal (draft a post, reactivate a paused offer, etc.)
5. End with open-ended CTA.
6. {lang_note}
7. NO URLs. Operator-peer tone, not alarming.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "perf_spike":
        metric = payload.get("metric", "calls")
        delta_pct = payload.get("delta_pct", 0)
        driver = payload.get("likely_driver", "")
        baseline = payload.get("vs_baseline", "")

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} celebrating a performance spike.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Metric: {metric} spiked +{delta_pct*100:.0f}% over 7d (baseline {baseline})
- Likely driver: {driver}
- Active offers: {active_offers}
- Signals: {signals}

RULES:
1. Open with good news — the spike and likely reason.
2. Suggest how to capitalise now (double down, post, run a limited offer).
3. One clear CTA.
4. Keep energy positive but not hype.
5. {lang_note}
6. NO URLs.

Output ONLY the message body. No quotes. No markdown."""

    elif kind in ("festival_upcoming", "seasonal_beat"):
        festival = payload.get("festival", "upcoming festival")
        days_until = payload.get("days_until", "")
        date = payload.get("date", "")
        cat_relevance = payload.get("category_relevance", [])
        seasonal_beats = category.get("seasonal_beats", [])
        beat_notes = [b.get("note", "") for b in seasonal_beats[:2]]

        prompt = f"""Write a SHORT (≤90 words) WhatsApp message from Vera to {owner} about an upcoming {festival}.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Active offers: {active_offers}
- Seasonal notes: {beat_notes}
- Days until festival: {days_until}

RULES:
1. Open with owner name and the festival hook — specific date and days remaining.
2. Tie to a relevant offer from their catalog or a service that peaks around this festival.
3. Propose one concrete action (create a GBP post, run a limited deal, etc.).
4. Single binary YES/STOP CTA.
5. {lang_note}
6. NO URLs.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "ipl_match_today":
        match = payload.get("match", "IPL match")
        venue = payload.get("venue", "")
        match_time = payload.get("match_time_iso", "")
        is_weeknight = payload.get("is_weeknight", True)
        # Parse time
        time_str = ""
        if match_time:
            try:
                mt = datetime.fromisoformat(match_time)
                time_str = mt.strftime("%I:%M %p")
            except Exception:
                time_str = match_time
        day_context = "weeknight" if is_weeknight else "weekend"

        prompt = f"""Write a SHORT (≤90 words) WhatsApp message from Vera to {owner} about today's IPL match.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Active offers: {active_offers}
- Signals: {signals}

IPL MATCH:
- Match: {match}
- Venue: {venue}
- Time: {time_str}
- Day type: {day_context}

INSIGHT: Saturday/Sunday IPL matches typically reduce restaurant dine-in covers by ~12% as people watch at home. Weeknight matches can boost delivery by 18-25%.

RULES:
1. Open with owner name and the match details (specific teams, time).
2. If {day_context} == "weekend", warn against in-venue promo and suggest pivoting to delivery with their existing offer.
3. If weeknight, recommend pushing delivery/takeaway now.
4. Reference their active offer by name.
5. Single specific proposal (e.g., Swiggy banner, Insta story) with time cap ("10 min").
6. {lang_note}
7. NO URLs. Operator voice.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "")
        count = payload.get("occurrences_30d", 0)
        trend = payload.get("trend", "")
        quote = payload.get("common_quote", "")

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} about a rising review theme.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}

REVIEW SIGNAL:
- Theme: {theme.replace("_", " ")}
- Count in 30d: {count}
- Trend: {trend}
- Common customer quote: "{quote}"

RULES:
1. Name the theme clearly — merchants respect directness.
2. Give the count (e.g., "4 reviews mentioned…").
3. Acknowledge the customer quote naturally.
4. Offer one concrete fix (operational, or draft a response template).
5. End with open-ended CTA.
6. {lang_note}
7. Not alarming — collegial.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "milestone_reached":
        metric = payload.get("metric", "review_count")
        value_now = payload.get("value_now", "")
        milestone = payload.get("milestone_value", "")
        is_imminent = payload.get("is_imminent", False)

        prompt = f"""Write a SHORT (≤75 words) WhatsApp message from Vera to {owner} about an approaching milestone.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Metric: {metric.replace("_", " ")}
- Current: {value_now}, approaching {milestone}

RULES:
1. Celebrate the imminent milestone — make it feel like a moment.
2. Suggest one action to push them over the line quickly (e.g., ask happy regulars to review).
3. Mention why milestones matter (e.g., 150 reviews puts them in top 10% of peers).
4. Single CTA.
5. {lang_note}

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "active_planning_intent":
        topic = payload.get("intent_topic", "")
        last_msg = payload.get("merchant_last_message", "")

        # Build a topic-specific response
        extra_facts = ""
        if "corporate" in topic and "thali" in topic:
            extra_facts = f"Merchant runs {mname} in {locality}. Active offer: {active_offers}. Delivery orders 30d: {cust_agg.get('delivery_orders_30d', 'N/A')}. Locality has many offices."
        elif "kids_yoga" in topic or "yoga" in topic:
            extra_facts = f"Merchant: {mname}, {locality}. Active members: {cust_agg.get('total_active_members', 'N/A')}. Trial-to-paid rate: {cust_agg.get('trial_to_paid_pct', 'N/A')}."

        prompt = f"""Write a SHORT (≤120 words) WhatsApp message from Vera directly continuing the merchant's planning request.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Active offers: {active_offers}
- {extra_facts}

MERCHANT'S LAST MESSAGE: "{last_msg}"
PLANNING TOPIC: {topic}

RULES:
1. Skip preamble — go straight to the plan/draft. The merchant already committed.
2. Give a concrete, ready-to-use artifact (menu pricing, program structure, a draft message).
3. Include specific numbers (prices, quantities, timelines).
4. Keep it structured if a list is appropriate.
5. End with one CTA asking for edit/confirm.
6. {lang_note}
7. NO URLs.

Output ONLY the message body (the plan/draft). No outer quotes. No markdown bold unless naturally part of the draft."""

    elif kind in ("winback_eligible", "dormant_with_vera"):
        days_lapsed = payload.get("days_since_expiry") or payload.get(
            "days_since_last_merchant_message", 30
        )
        sub_status = merchant.get("subscription", {}).get("status", "unknown")
        lapsed_customers = cust_agg.get("lapsed_90d_plus") or cust_agg.get(
            "lapsed_180d_plus", 0
        )
        last_topic = payload.get("last_topic", "")

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} re-engaging them.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Subscription: {sub_status}
- Days dormant: {days_lapsed}
- Lapsed customers: {lapsed_customers}
- Last topic: {last_topic}
- Active offers: {active_offers}
- Signals: {signals}

RULES:
1. Re-engage without guilt. Start with what they're missing out on, not "you haven't replied."
2. Use a verifiable hook — lapsed customer count, a fresh seasonal trend, or a digest insight.
3. Make re-engagement feel easy — "just say yes" / "I've got something relevant."
4. Single binary CTA.
5. {lang_note}
6. Tone: warm, direct, collegial.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "renewal_due":
        days_remaining = payload.get("days_remaining", 12)
        plan = payload.get("plan", "Pro")
        amount = payload.get("renewal_amount", "")

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} about subscription renewal.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Subscription: {plan}, {days_remaining} days remaining
- Renewal amount: ₹{amount}
- Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, CTR={my_ctr:.1%}
- CTR vs peer: {ctr_vs_peer}

RULES:
1. Open with the days remaining — make it specific.
2. Anchor the value: reference their actual performance numbers.
3. Suggest one thing they should do before renewing to show value (post, offer, etc.).
4. Binary YES/STOP CTA.
5. {lang_note}
6. Warm but urgent.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "gbp_unverified":
        uplift = payload.get("estimated_uplift_pct", 0.30)
        verify_path = payload.get("verification_path", "postcard or phone")

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} about unverified Google Business Profile.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Performance: views={perf.get('views', '?')}, calls={perf.get('calls', '?')}
- Est. uplift from verification: {uplift*100:.0f}%
- Verification path: {verify_path}

RULES:
1. Lead with the uplift — {uplift*100:.0f}% more visibility is the hook.
2. Explain the verification path briefly (postcard or phone call, takes ~7 days).
3. Offer to walk them through it.
4. Binary YES/STOP CTA.
5. {lang_note}

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "supply_alert":
        alert_id = payload.get("alert_id", "")
        molecule = payload.get("molecule", "")
        batches = payload.get("affected_batches", [])
        mfr = payload.get("manufacturer", "")
        chronic_count = cust_agg.get("chronic_rx_count", 0)
        # Estimate affected: ~9% of chronic customers (rough ratio from case study)
        affected_est = max(1, round(chronic_count * 0.09)) if chronic_count else ""

        prompt = f"""Write a SHORT (≤90 words) URGENT WhatsApp message from Vera to {owner} about a drug recall/supply alert.

PHARMACY FACTS:
- Name: {mname}, {locality}
- Chronic Rx customer count: {chronic_count}
- Estimated affected customers: {affected_est}

ALERT:
- Molecule: {molecule}
- Batches: {', '.join(batches)}
- Manufacturer: {mfr}
- Nature: voluntary recall (sub-potency, no safety risk, but needs replacement)

RULES:
1. Open with "urgent:" (lowercase). State molecule + batch numbers.
2. Mention no safety risk but replacement needed — precise language.
3. Give the derived count of likely affected customers.
4. Offer the complete workflow (patient WhatsApp note + pickup process).
5. Binary YES CTA (just reply YES to proceed).
6. {lang_note}
7. Trustworthy-precise tone. No alarm, no minimize.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "chronic_refill_due":
        molecules = payload.get("molecule_list", [])
        last_refill = payload.get("last_refill", "")
        runs_out = payload.get("stock_runs_out_iso", "")
        delivery_saved = payload.get("delivery_address_saved", False)
        cname = _safe(customer, "identity", "name") or "the customer"
        lang_pref = _safe(customer, "identity", "language_pref") or "hi"
        is_senior = _safe(customer, "identity", "senior_citizen") or False
        # Offers
        delivery_offer = next(
            (
                o
                for o in merchant.get("offers", [])
                if "delivery" in o.get("title", "").lower()
            ),
            None,
        )
        senior_offer = next(
            (
                o
                for o in merchant.get("offers", [])
                if "senior" in o.get("title", "").lower() or "15%" in o.get("title", "")
            ),
            None,
        )
        # Format runout date
        runout_str = ""
        if runs_out:
            try:
                ro = datetime.fromisoformat(runs_out.replace("Z", ""))
                runout_str = ro.strftime("%d %B")
            except Exception:
                runout_str = runs_out

        prompt = f"""Write a SHORT (≤100 words) WhatsApp message FROM {mname} TO {cname}'s household for a chronic medicine refill reminder.

CUSTOMER FACTS:
- Name: {cname}
- Language preference: {lang_pref}
- Senior citizen: {is_senior}
- Medicines due: {', '.join(molecules)}
- Stock runs out: {runout_str}
- Delivery address saved: {delivery_saved}

ACTIVE OFFERS: {[o['title'] for o in merchant.get('offers',[]) if o.get('status')=='active']}

RULES:
1. {'Use "Namaste" and respectful address (Sharma ji).' if is_senior else 'Friendly opening.'}
2. List all molecule names explicitly.
3. State the runout date.
4. Mention total cost with senior discount applied if available.
5. Offer free home delivery to saved address by specific time ("by 5pm tomorrow").
6. Binary: Reply CONFIRM to dispatch, or call to change dosage.
7. {lang_pref == 'hi' and 'Full Hindi.' or 'Hindi-English mix.' if 'hi' in lang_pref else 'English.'}
8. Trustworthy, precise tone.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "cde_opportunity":
        dig_item = _digest_item(category, payload.get("digest_item_id", "")) or {}
        title = dig_item.get("title", "CDE webinar")
        source = dig_item.get("source", "")
        cde_date = dig_item.get("date", "")
        credits = payload.get("credits", 2)
        fee = payload.get("fee", "")
        date_str = ""
        if cde_date:
            try:
                d = datetime.fromisoformat(cde_date)
                date_str = d.strftime("%d %b, %I:%M %p")
            except Exception:
                date_str = cde_date

        prompt = f"""Write a SHORT (≤75 words) WhatsApp message from Vera to {owner} about a CDE/webinar opportunity.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}

CDE:
- Title: {title}
- Date: {date_str}
- Credits: {credits}
- Fee: {fee}
- Source: {source}

RULES:
1. Open with owner name. Lead with the specific topic and credits.
2. Date + fee in one line.
3. Low-friction CTA ("Want the link?").
4. {lang_note}
5. Peer-collegial, not promotional.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "competitor_opened":
        comp_name = payload.get("competitor_name", "a new competitor")
        dist = payload.get("distance_km", "")
        comp_offer = payload.get("their_offer", "")
        opened = payload.get("opened_date", "")

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner} about a new nearby competitor.

MERCHANT FACTS:
- Name: {mname}, {locality}
- CTR: {my_ctr:.1%} (peer: {peer_ctr:.1%})
- Active offers: {active_offers}
- Review themes (positive): {[r for r in review_themes if r.get('sentiment')=='pos']}

COMPETITOR:
- Name: {comp_name}
- Distance: {dist} km
- Their offer: {comp_offer}
- Opened: {opened}

RULES:
1. No panic — collegial awareness tone.
2. State the competitor's offer and proximity.
3. Contrast with merchant's strength (reviews, service quality, active offers).
4. Suggest one strategic response (counter-offer, double down on reviews, etc.).
5. CTA: open-ended.
6. {lang_note}

Output ONLY the message body. No quotes. No markdown."""

    elif kind in ("customer_lapsed_hard", "customer_lapsed_soft"):
        cname = _safe(customer, "identity", "name") or "the customer"
        lang_pref = _safe(customer, "identity", "language_pref") or "english"
        days_lapsed = payload.get("days_since_last_visit", 60)
        focus = payload.get("previous_focus", "")
        months_member = payload.get("previous_membership_months", "")
        slot_pref = _safe(customer, "preferences", "preferred_slots") or ""
        # Find a relevant offer
        offer_str = active_offers[0] if active_offers else ""

        prompt = f"""Write a SHORT (≤90 words) WhatsApp message FROM {mname} TO lapsed member {cname}. Send_as: merchant_on_behalf.

CUSTOMER FACTS:
- Name: {cname}
- Days since last visit: {days_lapsed}
- Previous focus: {focus}
- Previous membership: {months_member} months
- Slot preference: {slot_pref}
- Language: {lang_pref}

ACTIVE OFFER: {offer_str}

RULES:
1. Open warmly — no shame, no guilt. Use "{days_lapsed // 7} weeks" or similar human framing.
2. Reference their training focus ({focus}) with a new relevant hook (new class, new slot).
3. Low-barrier offer — free trial or no-commitment slot.
4. Binary YES CTA. Emphasise "no auto-charge, no commitment."
5. {'Use Hindi-English mix.' if 'hi' in lang_pref else 'English only.'}
6. Warm, coaching tone. Short sentences.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "wedding_package_followup":
        cname = _safe(customer, "identity", "name") or "the customer"
        lang_pref = _safe(customer, "identity", "language_pref") or "english"
        wedding_date = payload.get("wedding_date", "")
        trial_completed = payload.get("trial_completed", "")
        days_to_wedding = payload.get("days_to_wedding", "")
        next_step = payload.get("next_step_window_open", "")
        slot_pref = _safe(customer, "preferences", "preferred_slots") or "Saturday"
        offer_str = active_offers[0] if active_offers else ""

        prompt = f"""Write a SHORT (≤90 words) WhatsApp message FROM {mname} TO bridal customer {cname}. Send_as: merchant_on_behalf.

CUSTOMER FACTS:
- Name: {cname}
- Wedding date: {wedding_date} ({days_to_wedding} days away)
- Bridal trial completed: {trial_completed}
- Next step: {next_step.replace('_', ' ')}
- Preferred slot: {slot_pref}
- Language: {lang_pref}

SALON OFFER: {offer_str}

RULES:
1. Address {cname} with a wedding emoji 💍 or 💐.
2. Reference salon owner first name ({owner}).
3. Mention {days_to_wedding} days to the wedding — creates natural urgency.
4. Name the next step program and its structure (sessions, price).
5. Offer to block their preferred slot.
6. Single binary CTA.
7. Warm, personal tone. {'English only.' if lang_pref == 'english' else 'English ok.'}

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "trial_followup":
        cname = _safe(customer, "identity", "name") or "the customer"
        lang_pref = _safe(customer, "identity", "language_pref") or "english"
        trial_date = payload.get("trial_date", "")
        sessions = payload.get("next_session_options", [])
        slot_label = sessions[0].get("label", "") if sessions else ""
        offer_str = active_offers[0] if active_offers else ""

        prompt = f"""Write a SHORT (≤80 words) WhatsApp message FROM {mname} TO {cname} following up on their trial. Send_as: merchant_on_behalf.

CUSTOMER:
- Name: {cname}
- Trial date: {trial_date}
- Language: {lang_pref}

NEXT SESSION: {slot_label}
OFFER: {offer_str}

RULES:
1. Warm opening. Reference the trial.
2. Offer the specific next session date/time.
3. Mention the membership offer.
4. Binary YES/STOP CTA. "No commitment for now."
5. {'Use Tamil-English mix if natural.' if 'ta' in lang_pref else 'English ok.'}

Output ONLY the message body. No quotes."""

    elif kind == "category_seasonal":
        season = payload.get("season", "")
        trends = payload.get("trends", [])
        shelf_action = payload.get("shelf_action_recommended", False)
        trend_str = ", ".join(trends)

        prompt = f"""Write a SHORT (≤85 words) WhatsApp message from Vera to {owner} about seasonal category trends.

PHARMACY FACTS:
- Name: {mname}, {locality}
- Chronic Rx customers: {cust_agg.get('chronic_rx_count', 'N/A')}
- Active offers: {active_offers}

SEASON: {season}
TRENDING DEMAND: {trend_str}

RULES:
1. Name the season and 2-3 specific demand shifts (with % changes).
2. Suggest shelf/inventory action.
3. If delivery not set up, nudge toward that.
4. Single CTA.
5. {lang_note}
6. Trustworthy, advisory tone.

Output ONLY the message body. No quotes. No markdown."""

    elif kind == "curious_ask_due":
        ask_template = payload.get("ask_template", "what_service_in_demand_this_week")
        last_ask = payload.get("last_ask_at", None)
        question_map = {
            "what_service_in_demand_this_week": f"what service has been most asked-for this week at {mname}?",
            "what_is_your_top_seller": f"what's your top-selling item / service right now?",
            "what_do_customers_ask_most": "what question do customers most often ask before booking?",
        }
        the_question = question_map.get(
            ask_template, "what's top-of-mind for you this week?"
        )

        prompt = f"""Write a SHORT (≤70 words) WhatsApp message from Vera to {owner} asking an engagement question.

MERCHANT: {mname}, {locality}
QUESTION TO ASK: {the_question}

RULES:
1. Open with owner first name.
2. Ask the question naturally and directly — no preamble.
3. Offer a specific low-effort reward for answering (e.g., "I'll turn it into a GBP post + a 4-line WhatsApp reply template. Takes 5 min.").
4. Keep it conversational, not promotional.
5. {lang_note}
6. NO CTA button — just the question + reward offer.

Output ONLY the message body. No quotes. No markdown."""

    else:
        customer_json = json.dumps(customer) if customer else "None"
        category_digest = json.dumps(category.get("digest", []))
        
        prompt = f"""Write a SHORT (≤80 words) WhatsApp message from Vera to {owner}.
We have received a new trigger event that requires reaching out to the merchant.

MERCHANT FACTS:
- Name: {mname}, {locality}
- Category: {slug}
- Signals: {signals}
- Active offers: {active_offers}

NEW TRIGGER DATA:
- Trigger Kind: {kind}
- Payload JSON: {json.dumps(payload)}

ADDITIONAL CONTEXT (use ONLY if relevant to the trigger):
- Customer context: {customer_json}
- Category digest items: {category_digest}

RULES:
1. Analyse the TRIGGER PAYLOAD. Synthesize what it means for the merchant and explain WHY this message is being sent right now.
2. Ground your response. You MUST use at least one concrete fact, number, or entity from the JSON payload or the merchant facts. Do NOT invent or assume any metrics or details not present in the provided context.
3. Keep it highly specific to the provided payload. If it's a dip, mention the dip. If it's a trend, mention the trend.
4. End with one clear, relevant CTA based on the context.
5. {lang_note}
6. Peer tone. NO URLs.

Output ONLY the message body. No quotes. No markdown."""

    return system, prompt


# ── CTA selector ───────────────────────────────────────────────────────────


def _pick_cta(trigger_kind: str, customer: Optional[dict]) -> str:
    BINARY_KINDS = {
        "regulation_change",
        "renewal_due",
        "gbp_unverified",
        "festival_upcoming",
        "supply_alert",
        "customer_lapsed_hard",
        "customer_lapsed_soft",
        "winback_eligible",
        "wedding_package_followup",
        "trial_followup",
    }
    NONE_KINDS = {"curious_ask_due"}
    OPEN_KINDS = {
        "research_digest",
        "perf_dip",
        "perf_spike",
        "review_theme_emerged",
        "milestone_reached",
        "active_planning_intent",
        "cde_opportunity",
        "competitor_opened",
        "dormant_with_vera",
        "category_seasonal",
        "seasonal_perf_dip",
        "ipl_match_today",
    }
    if trigger_kind in BINARY_KINDS:
        return "binary_yes_no"
    if trigger_kind in NONE_KINDS:
        return "none"
    if customer and trigger_kind == "recall_due":
        return "multi_choice_slot"
    if customer and trigger_kind == "chronic_refill_due":
        return "binary_confirm_cancel"
    return "open_ended"


# ── Anti-repetition check ──────────────────────────────────────────────────


def _is_auto_reply(text: str) -> bool:
    AUTO_PHRASES = [
        "thank you for contacting",
        "our team will respond",
        "automated assistant",
        "ek automated",
        "we'll get back",
    ]
    tl = text.lower()
    return any(p in tl for p in AUTO_PHRASES)


# ── Main compose function ──────────────────────────────────────────────────


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Returns:
        body              — WhatsApp message body
        cta               — CTA type string
        send_as           — "vera" or "merchant_on_behalf"
        suppression_key   — from trigger
        rationale         — short reasoning string
    """
    kind = trigger.get("kind", "unknown")
    slug = category.get("slug", "default")
    owner = _owner_first(merchant)
    mname = _merchant_name(merchant)
    suppression_key = trigger.get(
        "suppression_key", f"{kind}:{merchant.get('merchant_id','?')}"
    )
    send_as = "merchant_on_behalf" if customer else "vera"
    cta = _pick_cta(kind, customer)

    system, prompt = _build_prompt_for_trigger(category, merchant, trigger, customer)

    try:
        body = _llm(prompt, system)
        # Sanitise: strip leading/trailing quotes if LLM added them
        body = body.strip().strip('"').strip("'").strip()
    except Exception as e:
        # Fallback to a minimal deterministic body
        body = (
            f"Hi {owner}, quick note from Vera — "
            f"there's a {kind.replace('_',' ')} signal relevant to {mname}. "
            f"Want me to take a look and draft next steps?"
        )

    # Rationale
    perf = merchant.get("performance", {})
    rationale = (
        f"Trigger: {kind} | Merchant: {mname}, {slug} | "
        f"Signals: {merchant.get('signals', [])} | "
        f"CTR: {perf.get('ctr','?')} | Customer: {customer.get('identity',{}).get('name','none') if customer else 'none'} | "
        f"CTA: {cta} | Send-as: {send_as}"
    )

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": rationale,
    }
