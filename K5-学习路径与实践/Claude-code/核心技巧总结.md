视频详细介绍了 Claude Code 的多种高效用法和实用技巧。

以下是视频中提到的 Claude Code 主要用法和技巧总结：

**1.** **安装与集成**

- 安装 Claude Code 扩展，可用于 VS Code、Cursor 及 Windsurf 等 IDE，方便在 IDE 内直接启动 Claude Code。
- 支持多窗口/多面板并行运行，适合同时处理不同文件或代码库的不同部分。

**2.** **基本操作与界面**

- Claude Code 采用终端 UI，支持文件标签、选择上下文、slash 命令（如 /model 切换模型）。
- 可以自动将当前打开的文件加入上下文。
- 支持 Vim 模式（适合 Vim 用户）。

**3.** **模型与命令**

- 常用 /model 命令切换 Opus 和 Sonnet 两种模型，Opus 更强但有额度限制，Sonnet 更经济。
- 推荐用 clear 命令清理历史，避免 token 浪费和 LLM 自动摘要带来的延迟。
- 上箭头可回溯历史对话，包括跨会话内容。

**4.** **权限与自动化**

- 默认需要手动同意编辑文件或运行命令，可用 claude dangerously skip permissions（类似 yolo mode）跳过权限确认。
- 可通过自定义 hook 限制危险命令（如 rm -f）。

**5.** **代码审查与** **GitHub** **集成**

- 安装 GitHub app 后，Claude 可自动审查 PR，查找 bug 和安全隐患，支持自定义 code review prompt 只关注 bug 和漏洞。
- 能自动处理 PR 评论、根据反馈推送 commit。

**6.** **大型代码库支持**

- Claude Code 能稳定处理超大文件（如 1.8 万行 React 组件），比 Cursor 更少出错和卡顿。
- 擅长在大型代码库中搜索、理解组件关系和全局状态。

**7.** **队列与多任务**

- 支持消息队列（cue），可提前输入多个任务，Claude 会智能依次处理，极大提升多任务效率。

**8.** **自定义命令与记忆**

- 支持自定义 slash 命令和 hook，直接用自然语言描述即可。
- 支持项目级、目录级、全局级 memory（如 #always use mui components），自动保存到最相关的 cloud.md 文件。

**9.** **终端操作小技巧**

- 拖拽文件可自动引用，粘贴图片需用 ctrl+v（Mac）。
- 停止 Claude 需按 escape，双击 escape 可快速跳转历史消息。
- shift+tab 可自动接受权限请求。

**10.** **可视化界面与协作**

- 可用 builder.io VS Code/cursor/windsurf 扩展获得可视化 UI，支持实时预览、设计模式（Figma 风格）、浏览器端协作等。
- 支持直接在 PR 留言与 Claude 交互，自动响应并推送代码。

**11.** **价格与性价比**

- 推荐 Max 订阅计划，性价比高，远低于人工开发成本。

这些用法和技巧覆盖了 Claude Code 的安装、操作、自动化、协作、定制和效率提升等方方面面，适合开发者日常高效开发和团队协作。

Please review this pull request and look for bugs and security issues. Only report on bugs and potential vulnerabilities you find. Be concise.