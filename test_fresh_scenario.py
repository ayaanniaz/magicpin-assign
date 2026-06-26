import os, json, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['GEMINI_API_KEY'] = open('.env').read().split('=')[1].strip()

# Load real dataset for category and merchant
cat_r = json.load(open('dataset/categories/restaurants.json'))
merchants = json.load(open('dataset/merchants_seed.json'))['merchants']

# Get Pizza Junction (M005 is index 4 typically, let's just find it)
merchant = next(m for m in merchants if m['merchant_id'] == 'm_005_pizzajunction_restaurant_delhi')

# Create a COMPLETELY FRESH, UNSEEN TRIGGER
fresh_trigger = {
    "trigger_id": "trg_fresh_001",
    "merchant_id": merchant["merchant_id"],
    "scope": "merchant",
    "kind": "inventory_spoilage_risk", # Unseen kind
    "payload": {
        "item": "Fresh Mozzarella Cheese",
        "quantity_kg": 15.5,
        "expires_in_hours": 18,
        "financial_risk_inr": 6200,
        "recommended_action": "Run a 1-hour flash sale with 'Double Cheese' free upgrade to clear stock."
    },
    "created_at": "2026-06-26T10:00:00Z"
}

from composer import compose

print("=== TESTING FRESH SCENARIO: INVENTORY SPOILAGE RISK ===")
print("Sending an unseen trigger kind with a completely novel payload to the composer...\n")

result = compose(cat_r, merchant, fresh_trigger, customer=None)

print(f"MESSAGE BODY:\n{result['body']}")
print(f"\nCTA TYPE: {result['cta']}")
print(f"SEND_AS: {result['send_as']}")
