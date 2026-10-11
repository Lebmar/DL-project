# 四页主体英文工作稿

这是根据已完成 Stage2/Stage3 结果写成的学生审核草稿，不是已提交的课程报告。主体为第1–4页，参考文献与复现/作者说明为第5页；使用与 proposal 一致、未改动的官方 NeurIPS 2026 preprint 模板，没有缩小正文或改变页边距。

数据读数来自冻结的 outputs/stage3_analysis/。scripts/render_report_readout.py 只渲染已保存 CSV 的表格、数字与散点，不训练、选择模型、生成测试数据或重抽 bootstrap。readout_manifest.json 记录所有输入和渲染产物 SHA256；manifest.json 记录稿件与 PDF 的来源链和版式核查。

## 阅读入口

- [PDF 工作稿](../output/pdf/working_report.pdf)
- [可编辑 LaTeX](working_report.tex)
- [中文返来阅读说明](../research/current_status_zh.md)
- [完整 Stage3 结果与限制](../research/stage3_results.md)

稿件保留 entry 的部分重复与 recovery 的失败重复，区分固定模型的路径区间和训练数据/seed 敏感性；通道分解不被解释为唯一因果中介或通用架构优势。绝对覆盖率、宽度和分数表是原始预设参考情景的描述，不是新增主检验。AI 已获得课程允许，学生仍须理解、修订并承担个人作者责任；提交前应替换为规定的正式姓名。

## 重建

从仓库根目录运行渲染，再在 report/ 中编译；需要项目 Python 依赖及本地 LaTeX。这会覆盖本目录展示文件和 PDF，不改变冻结实验结果。

    .venv/bin/python scripts/render_report_readout.py
    cd report
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf working_report.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf working_report.tex

本次最新 PDF 五页均已渲染并逐页视觉检查：图例、表格、公式、页码和参考文献清楚，无正文溢出或未定义引用。后续修改后应重新编译、检查主体页数并更新 manifest；不要把旧 PDF 或旧 provenance 当新稿件的核查证据。
