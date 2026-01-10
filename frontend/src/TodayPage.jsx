import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

/** 建議：跟 ClosetPage 同一份選單，避免前後不一致 */
const CATEGORY_OPTIONS = [
  "blouse","cardigan","coat","dress","hoodie","jacket","jeans","leggings",
  "pants","shirt","shorts","skirt","sweater","t-shirt","top","vest"
]

const COLOR_OPTIONS = [
  "beige","black","blue","brown","burgundy","cream","gold","gray","green","grey",
  "ivory","khaki","maroon","navy","olive","orange","pink","purple","red","rose",
  "silver","tan","white","yellow"
]

function safeLower(s) {
  return (s ?? '').toString().trim().toLowerCase()
}

function itemImage(it) {
  return it?.image_url || it?.image || "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"
}

/** ✅ demo 用：簡單規則相似度（你隊友接模型後可整段換掉） */
function scoreSimilarity(candidate, closetItem) {
  let score = 0

  // category match（很重要）
  if (candidate.category && closetItem.category && candidate.category === closetItem.category) score += 0.55

  // color match（次重要）
  if (candidate.color && closetItem.color && candidate.color === closetItem.color) score += 0.25

  // title keyword overlap（小加分）
  const a = safeLower(candidate.title).split(/\s+/).filter(Boolean)
  const b = safeLower(closetItem.title).split(/\s+/).filter(Boolean)
  if (a.length && b.length) {
    const setB = new Set(b)
    const hit = a.filter(w => setB.has(w)).length
    score += Math.min(0.20, hit * 0.10)
  }

  return Math.max(0, Math.min(1, score))
}

export default function TodayPage({ go, user }) {
  // ====== closet load ======
  const [closet, setCloset] = useState([])
  const [loadingCloset, setLoadingCloset] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!user?.id) {
      setCloset([])
      setLoadingCloset(false)
      return
    }

    let alive = true
    async function loadCloset() {
      setLoadingCloset(true)
      setError('')

      const { data, error } = await supabase
        .from('closet_items')
        .select('id,title,category,color,worn,image_url,created_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })

      if (!alive) return
      if (error) setError(error.message)
      setCloset(data || [])
      setLoadingCloset(false)
    }

    loadCloset()
    return () => { alive = false }
  }, [user?.id])

  // ====== candidate form ======
  const [title, setTitle] = useState('')
  const [category, setCategory] = useState(CATEGORY_OPTIONS[0])
  const [color, setColor] = useState(COLOR_OPTIONS[0])
  const [imageUrl, setImageUrl] = useState('')
  const [preview, setPreview] = useState('')
  const [file, setFile] = useState(null)

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const previewSrc = preview || imageUrl

  // ====== analysis result ======
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null) // { decision, maxSim, reasons[], top[] }

  const closetCount = closet.length

  const topSimilar = useMemo(() => {
    if (!result?.top) return []
    return result.top
  }, [result])

  /** ✅ demo：先用 heuristic 分析；之後你隊友把這裡換成呼叫模型 */
  async function analyzeHeuristic() {
    if (!user?.id) return alert('請先登入才能分析')
    if (!closetCount) return alert('你的衣櫃目前是空的，先新增幾件衣服才好比對喔')

    const cand = {
      title: safeLower(title) || 'new item',
      category,
      color,
    }

    // 計算每一件衣櫃衣服的相似度
    const scored = closet.map((it) => ({
      ...it,
      sim: scoreSimilarity(cand, it),
    }))
    scored.sort((a, b) => b.sim - a.sim)

    const maxSim = scored[0]?.sim ?? 0
    const top = scored.slice(0, 3)

    // 決策（你可以調門檻）
    let decision = '可以買 ✅'
    if (maxSim >= 0.75) decision = '建議不要買 ⛔'
    else if (maxSim >= 0.55) decision = '看情況（很相似）⚠️'

    const reasons = []
    if (maxSim >= 0.75) reasons.push('與衣櫃中某件衣服高度相似（類別/顏色/文字描述重疊）')
    if (maxSim >= 0.55 && maxSim < 0.75) reasons.push('相似度偏高，可能是重複購入，建議先想搭配需求')
    if (maxSim < 0.55) reasons.push('衣櫃中找不到非常接近的款式，補齊衣櫃缺口的機會較高')

    // 額外：把最相似那件「穿著次數」拿來解釋（demo 很加分）
    const best = top[0]
    if (best) {
      if ((best.worn ?? 0) <= 1) reasons.push(`最相似的衣服「${best.title}」很少穿（worn=${best.worn ?? 0}），代表你可能不需要再買同款`)
      else reasons.push(`最相似的衣服「${best.title}」常穿（worn=${best.worn ?? 0}），若你想要替換/備用可考慮`)
    }

    setResult({ decision, maxSim, reasons, top })
  }

  /** ✅ 之後接模型：把 analyzeHeuristic 換成這種形式就行（先留註解）
   *
   * const { data, error } = await supabase.functions.invoke('should-buy', {
   *   body: { title, category, color, imageUrl, /* 或上傳後的 public url */ /* },
   * })
   * setResult(data)
   */

  return (
    <Shell
      go={go}
      title="買衣服建議"
      subtitle="上傳/輸入你想買的衣服，系統會和你的衣櫃比對相似度，建議你要不要買。"
    >
      {/* 工具列：回首頁 */}
      <div className="toolbar toolbarRow">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <div className="spacer" />
        <div style={{ opacity: 0.75, fontSize: 14 }}>
          衣櫃件數：{loadingCloset ? '讀取中...' : closetCount}
        </div>
      </div>

      {error && (
        <div style={{ marginTop: 10, padding: 10, border: '1px solid rgba(139,46,46,.35)', borderRadius: 12 }}>
          <strong style={{ color: '#8b2e2e' }}>Error：</strong> {error}
        </div>
      )}

      {/* ===== 表單卡 ===== */}
      <div className="card" style={{ marginTop: 14 }}>
        {previewSrc ? (
          <img className="cardImg" alt="candidate" src={previewSrc} />
        ) : (
          <div
            style={{
              height: 180,
              display: 'grid',
              placeItems: 'center',
              background: '#f4f2ef',
              color: 'rgba(74,44,29,0.7)',
              fontSize: 14
            }}
          >
            （可上傳照片或貼圖片網址）
          </div>
        )}

        <div className="cardBody">
          <div className="cardTopRow">
            <p className="cardTitle" style={{ margin: 0 }}>輸入想買的衣服</p>
            <span className="badge">Check</span>
          </div>

          <div className="formGrid" style={{ marginTop: 12 }}>
            <div className="field fieldFull">
              <label>上傳照片（可選）</label>
              <input type="file" accept="image/*" onChange={handleFile} />
            </div>

            <div className="field fieldFull">
              <label>圖片網址（備用，可留空）</label>
              <input
                className="control"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="貼上圖片 URL（可留空）"
              />
            </div>

            <div className="field">
              <label>名稱（可選）</label>
              <input
                className="control"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="例如：navy coat / white shirt"
              />
            </div>

            <div className="field">
              <label>類別</label>
              <select className="control" value={category} onChange={(e) => setCategory(e.target.value)}>
                {CATEGORY_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            </div>

            <div className="field">
              <label>顏色</label>
              <select className="control" value={color} onChange={(e) => setColor(e.target.value)}>
                {COLOR_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <div className="toolbar" style={{ marginTop: 14 }}>
            <button
              className="btn btnPrimary"
              disabled={busy || loadingCloset}
              onClick={async () => {
                setBusy(true)
                try {
                  await analyzeHeuristic()
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? '分析中...' : '開始分析'}
            </button>

            <button
              className="btn btnGhost"
              onClick={() => {
                setTitle('')
                setCategory(CATEGORY_OPTIONS[0])
                setColor(COLOR_OPTIONS[0])
                setImageUrl('')
                setPreview('')
                setFile(null)
                setResult(null)
              }}
            >
              清除
            </button>
          </div>
        </div>
      </div>

      {/* ===== 結果卡 ===== */}
      {result && (
        <div className="card" style={{ marginTop: 18 }}>
          <div className="cardBody">
            <div className="cardTopRow">
              <p className="cardTitle" style={{ margin: 0 }}>建議結果：{result.decision}</p>
              <span className="badge">
                Similarity {Math.round((result.maxSim || 0) * 100)}%
              </span>
            </div>

            <div className="meta" style={{ marginTop: 10 }}>
              {(result.reasons || []).map((r, idx) => (
                <span key={idx}>理由：{r}</span>
              ))}
            </div>

            <div style={{ marginTop: 14, fontWeight: 700 }}>
              最相似的衣櫃衣服（Top 3）
            </div>

            <div className="grid" style={{ marginTop: 10 }}>
              {topSimilar.map((it) => (
                <div key={it.id} className="card">
                  <img className="cardImg" alt={it.title} src={itemImage(it)} />
                  <div className="cardBody">
                    <div className="cardTopRow">
                      <p className="cardTitle">{it.title}</p>
                      <span className="badge">{Math.round((it.sim || 0) * 100)}%</span>
                    </div>
                    <div className="meta">
                      <span>{it.category}</span>
                      <span>{it.color}</span>
                      <span>穿過 {it.worn ?? 0} 次</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </Shell>
  )
}
