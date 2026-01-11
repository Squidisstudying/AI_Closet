import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# 設定 CORS 讓前端可以呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 1. 定義模型結構 (必須與訓練腳本一致)
# ===========================
class MultiHeadResNet(nn.Module):
    def __init__(self, num_cats, num_cols):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False) # 推理時不需要下載預訓練權重
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_cat = nn.Linear(num_features, num_cats)
        self.fc_color = nn.Linear(num_features, num_cols)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_cat(features), self.fc_color(features)

# ===========================
# 2. 設定標籤與載入模型
# ===========================
CAT_MAP = {
    0: 'Capris',
    1: 'Jackets',
    2: 'Jeans',
    3: 'Leggings',
    4: 'Shirts',
    5: 'Shorts',
    6: 'Skirts',
    7: 'Sweaters',
    8: 'Sweatshirts',
    9: 'Track Pants',
    10: 'Trousers',
    11: 'Tshirts',
    12: 'Tunics',
}

COLOR_MAP = {
    0: 'Black',
    1: 'Blue',
    2: 'Red',
    3: 'White',
    4: 'Grey Melange',
    5: 'Pink',
    6: 'Charcoal',
    7: 'Navy Blue',
    8: 'Grey',
    9: 'Beige',
    10: 'Yellow',
    11: 'Brown',
    12: 'Green',
    13: 'Purple',
    14: 'Turquoise Blue',
    15: 'Olive',
    16: 'Cream',
    17: 'Maroon',
    18: 'Peach',
    19: 'Teal',
    20: 'Lavender',
    21: 'Orange',
    22: 'Rust',
    23: 'Magenta',
    24: 'Nude',
    25: 'Sea Green',
    26: 'Mustard',
    27: 'Multi',
    28: 'Gold',
    29: 'Off White',
    30: 'Tan',
    31: 'Mauve',
    32: 'Khaki',
    33: 'Coffee Brown',
    34: 'Burgundy',
    35: 'Lime Green',
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadResNet(len(CAT_MAP), len(COLOR_MAP))

# 載入權重
try:
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")

# 定義圖片轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===========================
# 3. API 路由
# ===========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """辨識圖片的分類與顏色"""
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        p_cat, p_col = model(input_tensor)
        cat_idx = torch.argmax(p_cat).item()
        col_idx = torch.argmax(p_col).item()
    
    return {
        "category": CAT_MAP.get(cat_idx, "Unknown"),
        "color": COLOR_MAP.get(col_idx, "Unknown")
    }

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    既有的相似度比對功能 (示意，你可以保留原本的邏輯或整合 embedding)
    這裡簡單實作一個假的回傳，請替換成你原本的相似度邏輯
    """
    # 這裡應該要是你原本計算相似度的 code
    # 為了示範，這裡回傳一個假分數
    return {"similarity": 75.5} 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)