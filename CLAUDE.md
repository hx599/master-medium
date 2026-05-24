# master-medium 项目说明

## 项目背景
本目录是高光谱图像零样本分类（HZSCM）方向的研究工作区，包含中期实验报告、对比实验结果、雷达图绘制、报告修订脚本等。日常工作以**学术研究 + 论文写作**为主，配套了完整的 AI 写作 skill 链。

## Skills 总览（`.claude/skills/`，共 13 个）

按工作流分四组。所有 skill 均为本地文件夹（含 `SKILL.md`），Cursor / Claude Code 启动时自动发现。

### A. 研究 → 写作 → 评审 流水线（中文优先，原项目自带）

| Skill | 来源 | 用途 | 主要触发词 |
|---|---|---|---|
| `deep-research` | 本仓库自带 | 13-agent 深度研究：选题、文献检索、来源验证、综合 | `研究` `深度研究` `文献回顾` `引導我的研究` |
| `academic-paper` | 本仓库自带 | 12-agent 论文写作：6 种论文类型、5 种引用格式、双语摘要、LaTeX/DOCX/PDF | `寫論文` `學術論文` `審查意見` |
| `academic-paper-reviewer` | 本仓库自带 | 5 人模拟同行评审（主编 + 3 reviewer + Devil's Advocate） | `peer review` `referee report` `review my paper` |
| `academic-pipeline` | 本仓库自带 | 编排器，把上面三个串成 10 阶段闭环 + 强制诚信检查 | `academic pipeline` `research to paper` |

### B. ML / Systems 顶会论文（英文，来自 [zechenzhangAGI/AI-research-SKILLs](https://github.com/zechenzhangAGI/AI-research-SKILLs)）

| Skill | 用途 | 主要触发词 |
|---|---|---|
| `ml-paper-writing` | 投 NeurIPS / ICML / ICLR / ACL / AAAI / COLM 的完整论文写作；从 repo 起稿、LaTeX 模板、citation 验证（含 BibTeX）、reviewer checklist | `投 NeurIPS` `ICLR 模板` `Related Work` `paper checklist` |
| `systems-paper-writing` | 投 OSDI / SOSP / ASPLOS / NSDI / EuroSys 的 10–12 页系统论文，含段落级蓝图与 LaTeX 模板 | `OSDI` `SOSP` `systems paper` |
| `academic-plotting` | 论文配图：架构图（Gemini 生图）+ 数据图（matplotlib/seaborn），自动选图类型 | `画框架图` `bar chart` `ablation 图` |
| `presenting-conference-talks` | 从已编译论文生成 Beamer PDF + PPTX 演讲稿（含 speaker notes） | `oral talk` `slides` `spotlight` |

### C. 文档操作（来自 [anthropics/skills](https://github.com/anthropics/skills)）

| Skill | 用途 | 主要触发词 |
|---|---|---|
| `docx` | 创建/编辑/分析 .docx：模板填充、tracked changes、redlining；适合期刊 Word 投稿模板 | `Word 文档` `.docx` `修订痕迹` |
| `doc-coauthoring` | 三阶段协作写作流程：上下文收集 → 分节起草 → 读者测试 | `写文档` `proposal` `decision doc` |
| `canvas-design` | 先出 design philosophy（.md），再画布上实现为 .png/.pdf；做概念图、示意图 | `画海报` `design` `概念图` |

### D. 终稿打磨 + 元工具

| Skill | 来源 | 用途 | 主要触发词 |
|---|---|---|---|
| `humanizer` | [blader/humanizer](https://github.com/blader/humanizer) | 去 AI 味，基于 Wikipedia "Signs of AI writing" | `humanize` `去 AI 味` |
| `skill-creator` | 本仓库自带 | 造 skill 的 skill：创建/改进/跑 eval/优化 description | `create a skill` `improve skill` |

## 典型工作流

**写一篇中文学术报告** → `academic-pipeline`（自动调度 deep-research → academic-paper → academic-paper-reviewer）

**投英文顶会** → `ml-paper-writing` 起稿 + `academic-plotting` 配图 + `humanizer` 去 AI 味 + `presenting-conference-talks` 出 slides

**用期刊 Word 模板交稿** → `docx`

**做概念图/框架图** → `canvas-design` 出图 + `ml-paper-writing` 写 caption

**结构化写某一节** → `doc-coauthoring`

## 调用方式

- **自然语言触发**：直接说需求（如「帮我用 ICLR 2026 模板新建一篇」），相关 skill 会被加载
- **手动调用**：对话中输入 `/skill-name` 或在 Cursor / Claude Code 的 Skill 面板选择
- **查看 skill 内容**：`.claude/skills/<name>/SKILL.md`

## 来源与许可

| Skill | 上游仓库 | License |
|---|---|---|
| 4 个 academic-* + skill-creator + deep-research | 本仓库 | — |
| ml-paper-writing / systems-paper-writing / academic-plotting / presenting-conference-talks | `zechenzhangAGI/AI-research-SKILLs` | MIT |
| docx / doc-coauthoring / canvas-design | `anthropics/skills` | 见各 LICENSE.txt |
| humanizer | `blader/humanizer` | MIT |

## Claude Code 内置 skill（不在 `.claude/skills/`）
以下 skill 由 Claude Code CLI 直接提供，编译进二进制，无需安装：
`update-config` `keybindings-help` `simplify` `fewer-permission-prompts` `loop` `claude-api` `init` `review` `security-review`

会话中可直接 `/loop`、`/review` 等使用。
