import requests
from bs4 import BeautifulSoup
import json
import time
import re

# ==================================================
# 1. 基本設定（測試版）
# ==================================================

BASE_URL = "https://wear.jp"
START_PAGE = 1
END_PAGE = 80
MAX_OUTFITS_PER_PAGE = 25

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OutfitResearchBot/1.0",
    "Accept": "text/html"
}

OUTPUT_FILE = "wear_outfit_pairs_color_separated_test.json"

# ==================================================
# 2. 上衣 / 下裝 類型
# ==================================================

TOP_KEYWORDS = {
    "t-shirt": ["tシャツ", "t-shirt", "tee"],
    "shirt": ["シャツ", "shirt"],
    "hoodie": ["パーカー", "hoodie"],
    "sweater": ["ニット", "sweater"],
    "blouse": ["ブラウス", "blouse"]
}

BOTTOM_KEYWORDS = {
    "jeans": ["デニム", "jeans"],
    "wide pants": ["ワイドパンツ", "ワイド"],
    "slim pants": ["スリムパンツ", "スリム"],
    "flare pants": ["フレアパンツ", "フレア"],
    "pants": ["パンツ"]
}

# ==================================================
# 3. 細顏色（日文 → 英文）
# ==================================================

JP_COLORS_FINE = {
    "black": ["黒", "ブラック"],
    "white": ["白", "ホワイト"],
    "gray": ["グレー"],
    "light blue": ["ライトブルー"],
    "blue": ["青", "ブルー"],
    "navy": ["ネイビー"],
    "beige": ["ベージュ"],
    "brown": ["ブラウン", "茶"],
    "green": ["グリーン", "緑"],
    "red": ["レッド", "赤"]
}

# ==================================================
# 4. 基礎抽取工具
# ==================================================

def extract_gender(text):
    if any(k in text for k in ["レディース", "women", "女性"]):
        return "female"
    if any(k in text for k in ["メンズ", "men", "男性"]):
        return "male"
    return None


def find_positions(text, keywords):
    positions = []
    for kw in keywords:
        for m in re.finditer(kw, text):
            positions.append(m.start())
    return positions


def find_color_positions(text):
    results = []
    for color, kws in JP_COLORS_FINE.items():
        for kw in kws:
            for m in re.finditer(kw, text):
                results.append((color, m.start()))
    return results

# ==================================================
# 5. 顏色分配（核心！）
# ==================================================

def assign_color(item_positions, color_positions):
    if not item_positions or not color_positions:
        return None

    min_dist = float("inf")
    chosen_color = None

    for item_pos in item_positions:
        for color, color_pos in color_positions:
            dist = abs(item_pos - color_pos)
            if dist < min_dist:
                min_dist = dist
                chosen_color = color

    return chosen_color

# ==================================================
# 6. 抓單一穿搭頁
# ==================================================

def crawl_outfit_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
    except requests.exceptions.RequestException:
        return None

    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True).lower()

    gender = extract_gender(text)

    top_type = None
    bottom_type = None
    top_positions = []
    bottom_positions = []

    for t, kws in TOP_KEYWORDS.items():
        pos = find_positions(text, kws)
        if pos:
            top_type = t
            top_positions = pos
            break

    for b, kws in BOTTOM_KEYWORDS.items():
        pos = find_positions(text, kws)
        if pos:
            bottom_type = b
            bottom_positions = pos
            break

    color_positions = find_color_positions(text)

    top_color = assign_color(top_positions, color_positions)
    bottom_color = assign_color(bottom_positions, color_positions)

    if gender and top_type and bottom_type:
        return {
            "gender": gender,
            "top": {
                "type": top_type,
                "color": top_color
            },
            "bottom": {
                "type": bottom_type,
                "color": bottom_color
            }
        }

    return None

# ==================================================
# 7. 主爬蟲（五欄位完全相同才去重）
# ==================================================

def crawl_wear():
    outfits = []
    seen = set()

    for page in range(START_PAGE, END_PAGE + 1):
        print(f"\n→ Crawling WEAR page {page}")
        r = requests.get(f"{BASE_URL}/coordinate/?pageno={page}", headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/") and href.count("/") >= 2:
                if any(x in href for x in ["shop", "search", "tag"]):
                    continue
                links.append(BASE_URL + href)

        links = list(dict.fromkeys(links))[:MAX_OUTFITS_PER_PAGE]
        print(f"  Found {len(links)} outfits")

        for i, url in enumerate(links, 1):
            print(f"    [{i}/{len(links)}] parsing")
            outfit = crawl_outfit_page(url)
            if not outfit:
                continue

            key = (
                outfit["gender"],
                outfit["top"]["type"],
                outfit["top"]["color"],
                outfit["bottom"]["type"],
                outfit["bottom"]["color"]
            )

            if key in seen:
                continue

            seen.add(key)
            outfits.append(outfit)
            time.sleep(0.2)

    return outfits

# ==================================================
# 8. 執行
# ==================================================

def main():
    data = crawl_wear()
    print(f"\nCollected {len(data)} unique outfit pairs")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
