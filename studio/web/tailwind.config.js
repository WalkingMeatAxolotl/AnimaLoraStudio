/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // ── 背景层级 ─────────────────────────────────────────────
        // 用法: bg-canvas  bg-surface  bg-sunken  bg-overlay  bg-elevated
        canvas:   'var(--bg-canvas)',
        surface:  'var(--bg-surface)',
        sunken:   'var(--bg-sunken)',
        overlay:  'var(--bg-overlay)',
        elevated: 'var(--bg-elevated)',

        // ── 前景文本 ─────────────────────────────────────────────
        // 用法: text-fg-primary  text-fg-secondary  text-fg-tertiary  ...
        'fg-primary':   'var(--fg-primary)',
        'fg-secondary': 'var(--fg-secondary)',
        'fg-tertiary':  'var(--fg-tertiary)',
        'fg-disabled':  'var(--fg-disabled)',
        'fg-inverse':   'var(--fg-inverse)',

        // ── 边框色 ───────────────────────────────────────────────
        // 用法: border-subtle(细)  border-dim(默认)  border-bold(粗)
        // 注意：不用 border-{border-*} 命名，避免前缀重复
        subtle: 'var(--border-subtle)',
        dim:    'var(--border-default)',
        bold:   'var(--border-strong)',

        // ── 强调色 ───────────────────────────────────────────────
        // 用法: bg-accent  text-accent  bg-accent-soft  text-accent-fg ...
        accent: {
          DEFAULT: 'var(--accent)',
          hover:   'var(--accent-hover)',
          soft:    'var(--accent-soft)',
          fg:      'var(--accent-fg)',
        },

        // ── 列表选中行 ───────────────────────────────────────────
        // 用法: bg-selected-soft border-selected（radio 选中行高亮；暗色
        // 下比 accent 系更暗淡低饱和，见 tokens.css --row-selected-*）
        selected: {
          DEFAULT: 'var(--row-selected-border)',
          soft:    'var(--row-selected-bg)',
        },

        // ── 状态色 ───────────────────────────────────────────────
        // 用法: bg-ok  text-ok  bg-ok-soft / bg-err  text-err  bg-err-soft / ...
        ok:   { DEFAULT: 'var(--ok)',   soft: 'var(--ok-soft)'   },
        warn: { DEFAULT: 'var(--warn)', soft: 'var(--warn-soft)' },
        err:  { DEFAULT: 'var(--err)',  soft: 'var(--err-soft)'  },
        info: { DEFAULT: 'var(--info)', soft: 'var(--info-soft)' },
      },

      // ── 字号 → CSS 变量，支持运行时密度调节 ──────────────────────
      // extend 中同名 key 会覆盖 Tailwind 默认值：
      //   text-xs   → var(--t-xs,  12px)
      //   text-sm   → var(--t-sm,  14px)
      //   text-base → var(--t-base,15px)  (Tailwind 默认 16px，这里改为 15px)
      //   text-lg   → var(--t-lg,  18px)  (与 Tailwind 默认一致)
      //   text-xl   → var(--t-xl,  22px)  (Tailwind 默认 20px)
      //   text-2xl  → var(--t-2xl, 28px)  (Tailwind 默认 24px)
      //   text-3xl  → var(--t-3xl, 36px)  (与 Tailwind 默认一致)
      // text-md 是我们扩展的，Tailwind 无此级别
      fontSize: {
        '2xs':  ['var(--t-2xs)', { lineHeight: '1.4' }],
        'xs':   ['var(--t-xs)',   { lineHeight: '1.5' }],
        'sm':   ['var(--t-sm)',   { lineHeight: '1.5' }],
        'base': ['var(--t-base)', { lineHeight: '1.6' }],
        'md':   ['var(--t-md)',   { lineHeight: '1.5' }],
        'lg':   ['var(--t-lg)',   { lineHeight: '1.4' }],
        'xl':   ['var(--t-xl)',   { lineHeight: '1.3' }],
        '2xl':  ['var(--t-2xl)', { lineHeight: '1.2'  }],
        '3xl':  ['var(--t-3xl)', { lineHeight: '1.15' }],
      },

      // ── 阴影 → CSS 变量，支持深色模式自动切换 ───────────────────
      // 用法: shadow-sm  shadow-md  shadow-lg  shadow-xl
      boxShadow: {
        sm: 'var(--sh-sm)',
        md: 'var(--sh-md)',
        lg: 'var(--sh-lg)',
        xl: 'var(--sh-xl)',
      },
    },
  },
  plugins: [],
}
