import requests
import json
import time
import re

# ==================================================
# 1. 基本設定
# ==================================================

SUBREDDITS = {
    "male": [
        "malefashionadvice",
        "streetwear"
    ],
    "female": [
        "femalefashionadvice",
        "OUTFITS"
    ]
}

POST_LIMIT = 500          # 每個 subreddit 最多抓幾篇
SLEEP = 1.0               # Reddit 請求間隔（不要太快）

HEADERS = {
    "User-Agent": "OutfitResearchBot/1.0 (academic project)"
}

OUTPUT_FILE = "reddit_outfit_pairs.json"

# ==================================================
# 2. 類型與顏色字典（英文）
# ==================================================

TOP_KEYWORDS = {
    "t-shirt": ["t-shirt", "tee"],
    "shirt": ["shirt"],
    "hoodie": ["hoodie"],
    "sweater": ["sweater", "knit"],
    "jacket": ["jacket", "coat"]
}

BOTTOM_KEYWORDS = {
    "jeans": ["jeans", "denim"],
    "wide pants": ["wide pants", "wide trousers"],
    "pants": ["pants", "trousers"],
    "shorts": ["shorts"],
    "skirt": ["skirt"]
}

COLORS = {
    "black": ["black"],
    "white": ["white"],
    "gray": ["gray", "grey"],
    "navy": ["navy"],
    "blue": ["blue"],
    "light blue": ["light blue"],
    "dark blue": ["dark blue"],
    "beige": ["beige"],
    "brown": ["brown"],
    "green": ["green"],
    "olive": ["olive"],
    "red": ["red"],
    "pink": ["pink"]
}

# ==================================================
# 3. 抽取工具
# ==================================================

def extract_category(text, mapping):
    for cat, kws in mapping.items():
        for kw in kws:
            if kw in text:
                return cat
    return None


def extract_all_colors(text):
    found = []
    for color, kws in COLORS.items():
        for kw in kws:
            if kw in text:
                found.append(color)
                break
    return list(dict.fromkeys(found))


# ==================================================
# 4. 解析一篇 Reddit 貼文
# ==================================================

def parse_post(text, gender):
    text = text.lower()

    top_type = extract_category(text, TOP_KEYWORDS)
    bottom_type = extract_category(text, BOTTOM_KEYWORDS)
    colors = extract_all_colors(text)

    if not top_type or not bottom_type or len(colors) == 0:
        return None

    # 顏色分配策略（穩定版）
    if len(colors) == 1:
        top_color = bottom_color = colors[0]
    else:
        top_color = colors[0]
        bottom_color = colors[1]

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

# ==================================================
# 5. 主爬蟲
# ==================================================

def crawl_reddit():
    outfits = []
    seen = set()

    for gender, subs in SUBREDDITS.items():
        for sub in subs:
            print(f"\n→ Crawling r/{sub}")

            url = f"https://www.reddit.com/r/{sub}/top.json?t=year&limit={POST_LIMIT}"
            r = requests.get(url, headers=HEADERS)

            if r.status_code != 200:
                print("  ✖ Failed")
                continue

            posts = r.json()["data"]["children"]
            print(f"  Found {len(posts)} posts")

            for post in posts:
                data = post["data"]
                text = (data.get("title", "") + " " + data.get("selftext", "")).strip()

                outfit = parse_post(text, gender)
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

            time.sleep(SLEEP)

    return outfits

# ==================================================
# 6. 執行
# ==================================================

def main():
    data = crawl_reddit()
    print(f"\nCollected {len(data)} outfit pairs")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
