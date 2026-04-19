from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, asyncio, json
from openai import AsyncOpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
ai = AsyncOpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "webpage.html")
inventory: dict = {}
event_log: list = []

class ConnectionManager:
    def __init__(self): self.active = []
    async def connect(self, ws):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        self.active.remove(ws)
    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.active.remove(ws)

manager = ConnectionManager()

def compute_status(count):
    if count <= 0: return "Empty"
    elif count == 1: return "Low"
    elif count == 2: return "Medium"
    else: return "Good"

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open(HTML_PATH, "r", encoding="utf-8") as f: return f.read()

@app.get("/inventory")
async def get_inventory(): return {"inventory": inventory, "events": event_log[-20:]}

@app.post("/update-vision")
async def update_vision(data: dict):
    item = data.get("item", "unknown").lower().strip()
    event_type = data.get("event", "").upper()
    if not event_type:
        event_type = "REMOVE" if data.get("status") == "Low" else "ADD"
    if item not in inventory: inventory[item] = {"count": 0, "status": "Good"}
    if event_type == "ADD": inventory[item]["count"] += 1
    elif event_type == "REMOVE": inventory[item]["count"] = max(0, inventory[item]["count"] - 1)
    inventory[item]["status"] = compute_status(inventory[item]["count"])
    log_entry = {"event": event_type, "item": item, "count": inventory[item]["count"]}
    event_log.append(log_entry)
    if len(event_log) > 100: event_log.pop(0)
    await manager.broadcast({"type": "inventory_update", "inventory": inventory, "last_event": log_entry})
    return {"status": "ok", "inventory": inventory[item]}

@app.post("/reset")
async def reset_inventory():
    inventory.clear(); event_log.clear()
    await manager.broadcast({"type": "inventory_update", "inventory": inventory, "last_event": None})
    return {"status": "reset"}

@app.get("/siri/low-stock")
async def siri_low_stock():
    low = [i for i, v in inventory.items() if v["status"] in ("Low", "Empty")]
    if not low: return {"reply": "Your fridge is fully stocked!"}
    return {"reply": f"You are running low on {', '.join(low)}."}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_json({"type": "inventory_update", "inventory": inventory, "last_event": None})
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: manager.disconnect(ws)

def _inventory_summary():
    if not inventory: return "The fridge is empty."
    return "\n".join(f"- {item}: {v['count']} unit(s) ({v['status']})" for item, v in inventory.items())

@app.get("/ai/recipes")
async def ai_recipes():
    summary = _inventory_summary()
    prompt = f"Here is what's currently in my fridge:\n{summary}\n\nSuggest 3 simple recipes I can make with these ingredients. For each recipe give: name, a one-line description, and which fridge items it uses. Be concise. Format as JSON array with keys: name, description, uses."
    resp = await ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], response_format={"type":"json_object"}, max_tokens=500)
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    recipes = data.get("recipes", data.get("items", list(data.values())[0] if data else []))
    return {"recipes": recipes, "inventory_snapshot": summary}

@app.get("/siri/recipe-check")
async def siri_recipe_check(dish: str = ""):
    if not dish: return {"reply": "Please tell me what dish you want to check."}
    summary = _inventory_summary()
    prompt = f"My fridge contains:\n{summary}\n\nThe user wants to make: {dish}\nDo they have the key ingredients? Reply in 1-2 short sentences suitable for Siri to read aloud. If missing ingredients, name them. Be direct and friendly."
    resp = await ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=120)
    return {"reply": resp.choices[0].message.content.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
