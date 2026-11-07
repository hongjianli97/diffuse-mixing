#!/usr/bin/env python3
"""Build the full Chinese report from reviewed documents; no numerical solves."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/project-20261107'
OUTPUT = ROOT / 'exports/LTN_project_complete_20261107.pdf'
SOURCES = ['docs/PROJECT_OVERVIEW.md', 'docs/THEORY_EVIDENCE_APPENDIX.md',
           'docs/CODE_GUIDE.md', 'docs/NEXT_STEPS.md']


def document(relative):
    source = ROOT / relative
    body = source.read_text()
    def link(match):
        label, target = match.group(1), match.group(2)
        if target.startswith(('http:', 'https:', '#')):
            return match.group(0)
        if match.start() and body[match.start()-1] == '!':
            path = (source.parent / target).resolve()
            return '[' + label + '](' + path.relative_to(ROOT).as_posix() + ')'
        path = (source.parent / target.split('#')[0]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(target)
        return '[' + label + '](https://github.com/hongjianli97/diffuse-mixing/blob/main/' + path.relative_to(ROOT).as_posix() + ')'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, body)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    front = '''---
title: "LTN 扩散与掺混"
subtitle: "理论主线、代码实现、验证证据与下一步工作"
author: "diffuse-mixing 项目整理"
date: "2026-11-07"
lang: zh-CN
toc-title: "目录"
---

本报告整理现有研究，不新增物理实验。Stage3 尚未通过，fluid ORC 关闭。
旧 v0.8.0 未恢复；当前软件检查、限定算例结果和一般物理主张分别说明。
用户已要求更新 GitHub；远端是否完成以交接回执为准。

\\newpage

'''
    merged = front + '\n\n\\newpage\n\n'.join(document(p) for p in SOURCES)
    md = REPORT / 'complete-report.md'
    tex = REPORT / 'complete-report.tex'
    md.write_text(merged)
    command = ['pandoc', str(md), '--from=markdown+tex_math_single_backslash',
               '--standalone', '--toc', '--toc-depth=2', '--top-level-division=section',
               '--lua-filter=' + str(REPORT / 'report-filter.lua'),
               '--include-in-header=' + str(REPORT / 'header.tex'),
               '--resource-path=' + str(ROOT), '-V', 'mainfont=Noto Serif CJK SC',
               '-V', 'sansfont=Noto Sans CJK SC', '-V', 'monofont=DejaVu Sans Mono',
               '-V', 'fontsize=10pt', '-V', 'papersize=a4', '-V', 'geometry:margin=20mm',
               '-V', 'colorlinks=true', '-V', 'urlcolor=blue', '-V', 'linkcolor=blue',
               '-o', str(tex)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    logs = result.stdout + result.stderr
    for _ in range(2):
        result = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                                 '-output-directory=' + str(REPORT), str(tex)],
                                cwd=ROOT, text=True, capture_output=True)
        logs += result.stdout + result.stderr
        (REPORT / 'build.log').write_text(logs)
        if result.returncode:
            raise RuntimeError('XeLaTeX failed: see reports/project-20261107/build.log')
    OUTPUT.write_bytes((REPORT / 'complete-report.pdf').read_bytes())
    inputs = SOURCES + ['reports/project-20261107/header.tex', 'reports/project-20261107/report-filter.lua',
                        'reports/project-20261107/figures/material-fv-space.png',
                        'reports/project-20261107/figures/density-errors.png']
    receipt = {'physical_solves': 0, 'source_documents': SOURCES,
               'input_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs},
               'pdf': OUTPUT.relative_to(ROOT).as_posix(),
               'pdf_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
    (REPORT / 'build-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
