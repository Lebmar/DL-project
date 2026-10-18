# 四页主体英文工作稿

更新于2026-10-18。正式署名为 Chengxuan YUAN (50025581)。这是根据已完成 Stage2/Stage3、分位数秩敏感性和公开 FX 扩展写成的学生审核草稿，尚未由助手提交。正文第1–4页，参考文献及复现说明第5页，附录第6–7页。使用未改动的官方 NeurIPS 2026 preprint 模板，未缩小正文字号或改变页边距。

- [工作稿 PDF](../output/pdf/working_report.pdf) · [LaTeX 源码](working_report.tex)
- [中文项目状态](../research/current_status_zh.md)
- [Stage3结果](../research/stage3_results.md) · [秩敏感性](../research/rank_sensitivity_results.md) · [FX扩展](../research/fx_extension_results.md)

Stage3 的原始读数与来源链保持不变。`scripts/render_report_readout.py`生成原Stage3图表；新的`scripts/render_report_extensions.py`读取rank与FX已保存CSV，核验其全部manifest输出哈希后生成预定表格。这些展示脚本不训练、不挑选模型、不重新bootstrap。`manifest.json`记录原PDF哈希、新稿来源、全部7页视觉检查和稿件边界；原稿在Git历史保留。

报告保留恢复阶段重复失败、训练随机性敏感性以及金融配置排序反转。Rank属于已看路径上的事后检查；FX属于已看历史上的探索，不能称为独立金融盲测。`J_FX`没有模拟中的无冲击反事实。公式、指标及正负结论都需要学生理解、评议和修订。

## 重建展示与 PDF

从仓库根目录运行，已有实验结果不变；这些展示命令会更新稿件派生文件和工作稿PDF：

```bash
.venv/bin/python scripts/render_report_readout.py
.venv/bin/python scripts/render_report_extensions.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf working_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf working_report.tex
```

本版已逐页渲染并视觉核对：公式、图例、表格、页码、引用正常，无overfull/underfull或未定义引用警告。后续源码变更需要重新编译和检查，不沿用旧核查状态。Proposal提交版与这份最终报告工作稿是不同文档，proposal本轮未修改。
