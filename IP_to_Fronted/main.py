# IP_to_Fronted/app.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import os

app = FastAPI()

# 允許前端跨域存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 移植你的模型結構 (來自 train_model_1.py)
# ==========================================
class MultiHeadResNet(nn.Module):
    """多頭 ResNet 模型 (分類別和顏色)"""
    def __init__(self, num_cats, num_cols):
        super().__init__()
        # 注意：訓練時 pretrained=True，推論時這裡設 True/False 其實沒差，因為會被 load_state_dict 覆蓋
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_cat = nn.Linear(num_features, num_cats)
        self.fc_color = nn.Linear(num_features, num_cols)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_cat(features), self.fc_color(features)

# ==========================================
# 2. 定義標籤 (順序必須與訓練資料集一致！)
# ==========================================
# ⚠️ 請根據你訓練時 styles.csv 跑出來的結果修改這裡
# 這裡暫時用你前端的列表，但順序很可能跟訓練時不同
CAT_LABELS = [
  "blouse","cardigan","coat","dress","hoodie","jacket","jeans","leggings",
  "pants","shirt","shorts","skirt","sweater","t-shirt","top","vest"
]

COLOR_LABELS = [
  "beige","black","blue","brown","burgundy","cream","gold","gray","green","grey",
  "ivory","khaki","maroon","navy","olive","orange","pink","purple","red","rose",
  "silver","tan","white","yellow"
]

# 設定模型參數 (如果不確定，執行 app.py 時看報錯訊息，它會告訴你權重檔裡的形狀是多少)
num_cats = len(CAT_LABELS)  # 或是直接填數字，例如 16
num_cols = len(COLOR_LABELS) # 或是直接填數字，例如 24

# ==========================================
# 3. 載入模型 (修正路徑版)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadResNet(num_cats, num_cols)

# 取得 app.py 目前所在的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 組合出權重檔的完整路徑
model_path = os.path.join(BASE_DIR, "model_weights.pth")

print(f"正在嘗試載入模型，路徑為: {model_path}") # 印出來檢查

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ 模型載入成功！")
except FileNotFoundError:
    print(f"❌ 仍然找不到檔案，請確認 {model_path} 是否真的存在。")
except RuntimeError as e:
    print(f"❌ 模型載入失敗 (尺寸不合): {e}")

# 圖片預處理 (需與訓練時一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}

    # 讀取並處理圖片
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        out_cat, out_col = model(img_tensor)
        
        # 取得最大機率的 index
        cat_idx = torch.argmax(out_cat, dim=1).item()
        col_idx = torch.argmax(out_col, dim=1).item()
        
        # 轉成文字標籤
        # 加上邊界檢查防止 index out of range
        pred_category = CAT_LABELS[cat_idx] if cat_idx < len(CAT_LABELS) else str(cat_idx)
        pred_color = COLOR_LABELS[col_idx] if col_idx < len(COLOR_LABELS) else str(col_idx)

    return {
        "category": pred_category,
        "color": pred_color,
        "success": True
    }