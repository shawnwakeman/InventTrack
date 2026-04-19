from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from bs4 import BeautifulSoup
import uvicorn
import os

app = FastAPI()

# Ensure this matches your file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "webpage.html")

def get_inventory_from_html():
    if not os.path.exists(HTML_PATH):
        return ["Error: HTML file missing"]

    with open(HTML_PATH, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    items = []
    # In your new HTML, the badge has the class "badge b-low"
    low_badges = soup.find_all("span", class_="b-low")
    
    for badge in low_badges:
        # Find the parent row to get the item name
        row = badge.find_parent("div", class_="t-row")
        if row:
            name_tag = row.find("div", class_="r-name")
            if name_tag:
                # Clean up text and remove emojis
                name = name_tag.get_text(strip=True).encode('ascii', 'ignore').decode('ascii')
                items.append(name.strip())
    
    return items

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/siri/low-stock")
async def siri_low_stock():
    low_items = get_inventory_from_html()
    
    if not low_items:
        # Fallback message so the Shortcut never gets "No Items"
        return {"reply": "Your fridge is fully stocked! Nothing is low right now."}
    
    items_sentence = ", ".join(low_items)
    return {"reply": f"You are running low on {items_sentence}."}

if __name__ == "__main__":
    # Use 0.0.0.0 so your iPhone (10.165.x.x) can connect
    uvicorn.run(app, host="0.0.0.0", port=8001)