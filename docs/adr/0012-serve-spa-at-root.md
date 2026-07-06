# 0012 — 前端 SPA 从 `/studio/` 子路径挂载改为根路径挂载

**状态**：Accepted
**日期**：2026-06-28
**决策者**：@WalkingMeatAxolotl

## 背景

历史上 Studio 前端 SPA 挂在 `/studio/` 子路径，裸根 `/` 用一个 302 跳到 `/studio/`：

- `vite.config.ts` 用 `base: '/studio/'`（资源 URL 烘成 `/studio/assets/...`）；
- `App.tsx` 的 react-router `basename: '/studio'`；
- `server.py` 把 SPA `app.mount("/studio", ...)`；
- `root.py` 的 `/` 路由 302 → `/studio/`。

这套子路径布局是早期为了给**根级兄弟页面**让路而选的——最显眼的是独立监控页
`monitor_smooth.html`（1051 行，QueueMonitor 用 iframe 嵌它）。SPA 如果当 catch-all
挂在根 `/`，会把这类根级页面一并吞掉，所以给 SPA 单开 `/studio/` 前缀来共存。

该理由在监控页被删那一刻就失效了：`faed9b77`（React 原生组件替换 iframe 监控）+
`d6b7ed25`（删 `monitor_smooth.html`、改指 SPA 内 `/tools/monitor`）之后，根级只剩
`/api/*`、`/samples/*` 和 FastAPI 内建 `/docs` 等**纯命名空间路由**，没有任何根级 HTML
页需要和 SPA 共存。`/studio/` 前缀失去功能依赖，`/` 退化成一个 vestigial 重定向。
`api/main.py` 那句「裸根路径只是兼容旧 monitor」其实也不准——根路径从来只是跳板，
旧监控页伺服在它自己的 `/monitor_smooth.html`，不在 `/`。

**触发本次决策的是 issue #330**：用户在 ModelScope 创空间（类 HuggingFace Space）用
Docker 反代容器 7860 端口运行 Studio。平台网关对 `/studio/` 做尾斜杠归一化，与
Starlette mount 固有的 307（`/studio` → `/studio/`）叠加成无限重定向环，UI 打不开。
这不是个例：HF Spaces、部分云 GPU 租赁机都是「反代容器端口到根」模式，而这恰恰是
本项目的真实部署场景。`/studio/` 子路径 + 重定向是这一整类平台的通用坑。

关键事实：前端的 API / 采样图调用全走 `/api`、`/samples`（根绝对路径），与 `/studio/`
前缀**解耦**；DB / 历史快照 / 输出里没有任何持久化的 `/studio/` URL；前缀只钉在上述
3 处接线点。

## 候选方案

- **A. 维持现状，关闭 issue。** 在 #330 下说明属平台部署适配、不在核心支持范围。
  优点：零改动。缺点：「反代容器端口」就是本项目的真实部署形态，坑会反复出现。
- **B. 加 `--base-path` 运行时可配置前缀。** 让用户按部署环境指定挂载前缀。
  缺点：Vite 的 `base` 是 **build-time 烘死**进资源 URL 的，运行时改前缀必须重建或在
  伺服时改写 index.html，复杂且违反「定死更好维护」。真正的需求是「回归标准布局」，
  不是「可配置」。
- **C（选定）. SPA catch-all 挂到根 `/`，API 留在 `/api`，删根重定向。** 这是 SPA + JSON
  API 的标准布局，根入口零重定向，顺手修复整类反代平台。

## 决策

采用方案 C：

1. `vite.config.ts`：`base: '/studio/'` → `base: '/'`。
2. `App.tsx`：react-router `basename` 由 `/studio` 改根（默认 `/`）。
3. `server.py`：`WEB_DIST` 存在时 `app.mount("/", SPA...)`；不存在时注册一个根路由返
   「请先构建前端」的 JSON 提示（原 dist-missing 行为搬家）。
4. `root.py`：删除 `/` → `/studio/` 重定向，改放**一次性 legacy 兼容跳转**
   `/studio` 与 `/studio/{rest}` → `/{rest}`（307，保留 query）。
5. `static.py`：`SPAStaticFiles` 加 guard——未命中路径的首段属于服务端命名空间
   （`api` / `samples`）时保持干净 404，不兜底 index.html。
6. CLI / 打印的入口 URL 由 `…/studio/` 改 `…/`（`cli.py`、`api/main.py`）。
7. 更新 `test_studio_server.py` 的根路径断言并重生成 route snapshot。

legacy `/studio/*` 兼容跳转保留**一个 release**，下版删除。

## 理由

- **否决 A**：部署场景决定坑会复现，不是个例；一次标准化换来整类平台可用，划算。
- **否决 B**：build-time base 让「可配置前缀」得不偿失；标准化到根更简单、更可维护，
  且彻底消除重定向（可配置前缀仍需重定向，仍可能踩网关归一化）。
- **选 C 的安全性**：非 `/api` 的服务端路由只有 `/` 和 `/samples/{filename}` 两条；
  显式路由全部注册在 SPA mount **之前**，exact match 优先，所以 `/api/*`、`/samples/*`、
  `/docs` 仍各自命中 handler，SPA fallback 只接管真正未知的非命名空间路径。唯一新增
  风险——未知 `/api/*`（末段无 `.`）被 index.html 兜底成 200——由 guard 关闭。
- **保留 legacy 跳转的理由**：本项目有 webui 自更新机制（`/api/system/update` → 重启）。
  老用户更新时浏览器 tab 仍停在 `/studio/...`，前端 reconnect 后会 `location.reload()`；
  没有兼容跳转这一下会 404，有则平滑落到 `/...`。因为能自更新，老 tab 必然撞上，所以
  这条跳转值得留一个 release。主入口 `/` 不跳转，故 ModelScope 等只打 `/` 的平台不受
  legacy 跳转影响。

## 后果

- **好处**：根入口零重定向；修复 #330 及整类「反代容器端口」平台（HF Spaces / 云 GPU
  租赁机）；布局回归标准 SPA + API，少一层认知负担。
- **新约束**：`SPAStaticFiles` 的 guard 维护一份「服务端根级命名空间」列表
  （当前 `api` / `samples`）。**将来新增非 `/api` 的根级路由必须同步进该列表**，否则
  其未命中子路径会被 SPA 兜底成 200。
- **待还的债**：legacy `/studio/*` 兼容跳转留一个 release 后删除；删除后老书签 / 深链
  （`…/studio/projects/…`）失效，需手动改访问根。
- **无数据迁移**：采样图走 `/samples`、接口走 `/api`，DB / 快照 / 输出无持久化
  `/studio/` URL，改前缀不碰任何存量数据。
- 正常用户走 `studio run` 由 CLI 自动开浏览器，改的只是开哪个 URL，无感。

## 参考

- issue #330（ModelScope 创空间 Docker 反代重定向环）
- `faed9b77` feat(monitor)：React 原生组件替换 iframe 监控
- `d6b7ed25` refactor：淘汰 `monitor_smooth.html`
- 改动文件：`studio/web/vite.config.ts`、`studio/web/src/App.tsx`、`studio/server.py`、
  `studio/api/routers/root.py`、`studio/api/static.py`、`studio/cli.py`、
  `studio/api/main.py`、`tests/test_studio_server.py`、`tests/_snapshots/studio_routes.json`
