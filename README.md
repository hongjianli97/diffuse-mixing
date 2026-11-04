# LTN 扩散与掺混 — Codex 交接包

2026-10-29。包含完整方法规范、代码审查、原始参考代码和可复现审查脚本。**这是整理后的研究交接包，不是已完成的新求解器。**

## 推荐阅读顺序

1. docs/METHOD.md：完整方法、质量/密度、两图一核、激波、扩散、动态 ORC 与验证要求。
2. docs/CODE_REVIEW.md：实际代码发现、复用决定和已执行验证。
3. docs/CODEX_TASK.md：可直接交给 Codex 的实施任务与接口。
4. docs/ARCHIVE_INDEX.md：每个压缩包及 Python 文件的来源映射。

## 目录

- reference_baseline/：完整恢复 v3.2 的代码/文本/历史结果目录；原始内容未修改。
- prototypes/：网络掺混 v2、优化、界面拉伸三组整理后的参考目录。
- legacy_sources/：所有归档中去重的代码与文本；按 audit 清单查找，不直接当独立可运行工程。
- audit/：归档/文件哈希清单、静态发现、依赖、notebook扫描、数值审查脚本及本次输出。
- requirements-reference.txt：参考代码依赖提示，不是经过全套测试的锁定环境。

包内未重复收录 PDF、图片、xlsx、原始 ZIP、pyc；原始成员仍记入清单。归档内没有 npz/npy/pkl/h5 原始算例数据；旧脚本常依赖外部 DNS 文件。动态 ORC 的可执行实现未在本批代码中找到，需要新建。

## 本次审查重跑

在本目录执行：

```bash
python audit/run_numeric_audit.py
```

已在 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0 运行。脚本只检查历史算子的有限案例和已发现反例，不证明新方法的连续一致性。符号测试因当前环境缺 sympy 未运行；如安装依赖，可在 reference_baseline 根目录运行 `python tests/test_identities_exact.py`，应读取逐项布尔结果，不能只看退出码。

历史 JSON 的目录为 results；本次新审查输出位于 audit，二者不要混用。所有保存的原始源码保留旧注释；其中不严谨的 exact、mass-consistent 等描述以 CODE_REVIEW 和 METHOD 的限定为准。
