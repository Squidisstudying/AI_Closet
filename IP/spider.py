import requests
import re
import json
import time

# =========================
# 1. 設定區
# =========================

SUBREDDITS = [
    "malefashionadvice",
    "femalefashionadvice"
]

POST_LIMIT = 500  # 每個 subreddit 抓幾篇

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Coding101 Outfit Project)"
}

# =========================
# 2. 衣物與顏色詞典
# =========================

CLOTHING = {
    "top": [
        "t-shirt", "tee", "shirt", "blouse",
        "hoodie", "sweater", "knit", "polo"
    ],
    "bottom": [
        "jeans", "pants", "trousers",
        "slacks", "skirt", "shorts"
    ],
    "outer": [
        "jacket", "coat", "blazer",
        "cardigan", "denim jacket"
    ],
    "shoes": [
        "sneakers", "shoes", "boots",
        "loafers", "converse"
    ]
}

COLORS = [
    "white", "black", "blue", "gray", "grey",
    "beige", "brown", "navy", "green", "red"
]

# =========================
# 3. 穿搭文字解析器
# =========================

def parse_outfit_text(text: str):
    text = text.lower()

    result = {
        "top": None,
        "bottom": None,
        "outer": None,
        "shoes": None,
        "colors": set()
    }

    for color in COLORS:
        if re.search(rf"\b{color}\b", text):
            result["colors"].add(color)

    for category, keywords in CLOTHING.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text):
                result[category] = kw
                break

    result["colors"] = list(result["colors"])
    return result

# =========================
# 4. 判斷是不是「像穿搭描述」
# =========================

def looks_like_outfit(text: str) -> bool:
    keywords = [
        "jeans", "shirt", "t-shirt", "pants",
        "jacket", "wearing", "outfit"
    ]
    text = text.lower()
    return any(k in text for k in keywords)

# =========================
# 5. 爬 Reddit 貼文
# =========================

def crawl_subreddit(subreddit: str, limit=50):
    url = f"https://www.reddit.com/r/{subreddit}/top.json?t=year&limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=10)

    if r.status_code != 200:
        print(f"Failed to fetch {subreddit}")
        return []

    data = r.json()
    posts = data["data"]["children"]

    texts = []
    for p in posts:
        title = p["data"].get("title", "")
        body = p["data"].get("selftext", "")
        combined = f"{title}. {body}".strip()
        if combined:
            texts.append(combined)

    return texts

# =========================
# 6. 主程式
# =========================

def main():
    outfits = []

    for sub in SUBREDDITS:
        print(f"Crawling r/{sub}...")
        texts = crawl_subreddit(sub, POST_LIMIT)

        for text in texts:
            parsed = parse_outfit_text(text)

            # 先放寬條件
            if parsed["top"] or parsed["bottom"]:
                outfits.append(parsed)

    print(f"Collected {len(outfits)} outfits")

    with open("outfits.json", "w", encoding="utf-8") as f:
        json.dump(outfits, f, ensure_ascii=False, indent=2)

    print("Saved to outfits.json")


if __name__ == "__main__":
    main()
