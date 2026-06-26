import os, json, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['GEMINI_API_KEY'] = open('.env').read().split('=')[1].strip()

cat_r = json.load(open('dataset/categories/restaurants.json'))
merchants = json.load(open('dataset/merchants_seed.json'))['merchants']
merchant = next(m for m in merchants if m['merchant_id'] == 'm_005_pizzajunction_restaurant_delhi')

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
        "recommended_action": "Run a 1-hour flash sale with 'Double Cheese' free upgrade to clear stock."
    }
}

from composer import _build_prompt_for_trigger, _get_client
from google.genai import types as genai_types

system, prompt = _build_prompt_for_trigger(cat_r, merchant, fresh_trigger, None)

_SAFETY_OFF = [
    genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
    genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
    genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="OFF"),
    genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="OFF"),
]

client = _get_client()
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,
        max_output_tokens=700,
        safety_settings=_SAFETY_OFF,
    ),
)
print("FINISH REASON:", resp.candidates[0].finish_reason if resp.candidates else "No candidates")
if resp.candidates:
    print("TEXT:", resp.text)
    print("SAFETY RATINGS:", resp.candidates[0].safety_ratings)
