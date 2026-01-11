import { useEffect, useState } from 'react'
import './App.css'
import { supabase } from './supabaseClient.js'

// 引入頁面組件
import MarketPage from './MarketPage.jsx'
import ClosetPage from './ClosetPage.jsx'
import TodayPage from './TodayPage.jsx' 
import AuthTest from './AuthTest.jsx'
import Shell from './Shell.jsx'

const CATEGORY_OPTIONS = [
  "capris", "jackets", "jeans", "leggings", "shirts", "shorts", "skirts", 
  "sweaters", "sweatshirts", "track pants", "trousers", "tshirts", "tunics"
]

export default function App() {
  // 預設回到首頁 ('home')
  const [page, setPage] = useState('home')

  // 登入狀態管理
  const [user, setUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)

  useEffect(() => {
    // 1) 先讀 session
    supabase.auth.getSession().then(({ data }) => {
      setUser(data.session?.user ?? null)
      setAuthLoading(false)
    })

    // 2) 監聽登入/登出（避免你登入後畫面不更新）
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    return () => {
      listener.subscription.unsubscribe()
    }
  }, [])

  // 載入中畫面
  if (authLoading) return <div style={{ padding: 20, textAlign: 'center' }}>Loading...</div>

  // 沒登入就先顯示登入畫面
  if (!user) return <AuthTest onLogin={setUser} />

  // 路由切換邏輯
  if (page === 'closet') return <ClosetPage go={setPage} user={user} />
  if (page === 'today') return <TodayPage go={setPage} user={user} />
  if (page === 'market') return <MarketPage go={setPage} user={user} />

  // 首頁畫面 (Home)
  return (
    <div className="home">
      <div className="homeInner">
        <TopNav variant="dark" go={setPage} />

        <div className="heroContent">
          <div className="heroBox">
            <h1 className="heroTitle">Dress smarter.</h1>
            <p className="heroSubtitle">
              管理衣櫃、購物建議、把不常穿的衣服一鍵上架二手交易。
            </p>

            <div className="heroActions">
              <button className="heroCardBtn" onClick={() => setPage('closet')}>
                進入我的衣櫃
              </button>
              <button className="heroCardBtn" onClick={() => setPage('today')}>
                買衣服建議
              </button>
              <button className="heroCardBtn" onClick={() => setPage('market')}>
                前往二手交易
              </button>
            </div>

            <p style={{ marginTop: 16, opacity: 0.7, fontSize: '0.9rem' }}>
              已登入：{user.email}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// 導覽列組件
function TopNav({ variant, go }) {
  const isLight = variant === 'light'
  return (
    <div
      className={`navbar ${isLight ? 'navbarLight' : ''}`}
      style={{ color: isLight ? '#4a2c1d' : '#fff' }}
    >
      <div className="brand" onClick={() => go('home')} style={{ cursor: 'pointer', fontWeight: 'bold', fontSize: '1.2rem' }}>
        My Style Closet
      </div>

      <div className="navMenu">
        <button className="navBtn" onClick={() => go('closet')}>我的衣櫃</button>
        <button className="navBtn" onClick={() => go('today')}>買衣服建議</button>
        <button className="navBtn" onClick={() => go('market')}>二手交易區</button>
      </div>
    </div>
  )
}