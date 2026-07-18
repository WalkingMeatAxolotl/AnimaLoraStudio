import { useEffect, useState } from 'react'

/** prompt 的真实 token 数（防抖调后端 tokenizer，与训练/推理同源）。
 *
 * 返回 null 表示不可用（端点失败 / tokenizer 未就绪 / 空文本），角标隐藏。
 * 纯信息展示——不做长度警告（超长截断与否是模型口径，质量由用户掌握）。
 */
export function useTokenCount(text: string, modelFamily: string): number | null {
  const [tokens, setTokens] = useState<number | null>(null)
  useEffect(() => {
    if (!text.trim()) {
      setTokens(null)
      return
    }
    let alive = true
    const timer = setTimeout(() => {
      fetch('/api/generate/token_count', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_family: modelFamily }),
      })
        .then((r) => r.json())
        .then((d: { tokens: number | null }) => {
          if (alive) setTokens(typeof d.tokens === 'number' ? d.tokens : null)
        })
        .catch(() => { if (alive) setTokens(null) })
    }, 500)
    return () => { alive = false; clearTimeout(timer) }
  }, [text, modelFamily])
  return tokens
}
