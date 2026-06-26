import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

# --- CONFIGURATION ---
# Replace this with your actual deployed URL (e.g., "https://your-bot.onrender.com")
# If testing locally, leave it as "http://localhost:8080"
BASE_URL = "https://magicpin-assign.onrender.com"
# ---------------------


def post_json(endpoint: str, data: dict):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error on {endpoint}: {e.code} - {e.read().decode()}")
        return None
    except urllib.error.URLError as e:
        print(f"Connection Error: {e.reason}")
        return None


def main():
    print(f"Testing against endpoint: {BASE_URL}\n")

    # 1. Load real data
    try:
        cat_r = json.load(open("dataset/categories/restaurants.json"))
        merchants = json.load(open("dataset/merchants_seed.json"))["merchants"]
        merchant = next(
            m
            for m in merchants
            if m["merchant_id"] == "m_005_pizzajunction_restaurant_delhi"
        )
    except Exception as e:
        print("Failed to load local dataset files:", e)
        return

    # 2. Push Category Context
    print("1. Pushing category context...")
    res = post_json(
        "/v1/context",
        {
            "scope": "category",
            "context_id": cat_r["slug"],
            "version": 1,
            "payload": cat_r,
        },
    )
    print(res)

    # 3. Push Merchant Context
    print("\n2. Pushing merchant context...")
    res = post_json(
        "/v1/context",
        {
            "scope": "merchant",
            "context_id": merchant["merchant_id"],
            "version": 1,
            "payload": merchant,
        },
    )
    print(res)

    # 4. Push Fresh Trigger Context
    print("\n3. Pushing fresh unseen trigger context...")
    fresh_trigger = {
        "trigger_id": "trg_fresh_001",
        "merchant_id": merchant["merchant_id"],
        "scope": "merchant",
        "kind": "inventory_spoilage_risk",
        "payload": {
            "item": "Fresh Mozzarella Cheese",
            "quantity_kg": 15.5,
            "expires_in_hours": 18,
            "financial_risk_inr": 6200,
            "recommended_action": "Run a 1-hour flash sale with 'Double Cheese' free upgrade to clear stock.",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    res = post_json(
        "/v1/context",
        {
            "scope": "trigger",
            "context_id": fresh_trigger["trigger_id"],
            "version": 1,
            "payload": fresh_trigger,
        },
    )
    print(res)

    # 5. Call /v1/tick to execute the engine
    print("\n4. Calling /v1/tick to process the trigger...")
    tick_payload = {
        "now": datetime.now(timezone.utc).isoformat(),
        "available_triggers": [fresh_trigger["trigger_id"]],
    }
    tick_res = post_json("/v1/tick", tick_payload)

    if not tick_res:
        return

    print("\n=== RESULT FROM SERVER ===")
    actions = tick_res.get("actions", [])
    if not actions:
        print("No actions returned.")
    for action in actions:
        print(f"\nACTION TYPE: {action.get('action')}")
        payload = action.get("payload", {})
        print(f"MESSAGE BODY:\n{payload.get('body')}")
        print(f"\nCTA: {payload.get('cta')}")
        print(f"RATIONALE: {payload.get('rationale')}")


if __name__ == "__main__":
    main()
