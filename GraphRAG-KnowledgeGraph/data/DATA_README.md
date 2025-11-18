# 数据集说明

## PDF 源文件

本项目使用 2 个 PDF 文件作为演示数据源：

1. **lesson6_1.pdf** (~1 MB)
   - 融资课程材料

2. **早期创业融资指南-奇绩创坛.pdf** (~2.3 MB)
   - 奇绩创坛创业融资指南

## 下载方式

**方式一：网盘下载**（推荐）

由于文件大小原因，未上传至 GitHub。请通过以下方式获取：

- 📥 **夸克网盘**: [11.17示例PDF](https://pan.quark.cn/s/c0fd289b3cae)

**方式二：使用自己的 PDF**

您也可以使用任意 PDF 文件替代，只需：

1. 将 PDF 文件放入 `data/pdfs/` 目录
2. 运行 `python src/parse_pdfs.py`
3. 后续流程保持不变

## 文件放置位置

```
GraphRAG-KnowledgeGraph/
└── data/
    └── pdfs/
        ├── lesson6_1.pdf
        └── 早期创业融资指南-奇绩创坛.pdf
```

## 数据处理流程

PDF 文件经过以下处理：

1. `parse_pdfs.py` → 提取文本块
2. `extract_triples.py` → LLM 抽取三元组
3. `import_to_neo4j.py` → 导入图数据库

处理后的中间文件已包含在项目中：

- `data/outputs/triples_output.json` - 抽取的三元组示例

## 版权说明

本项目 PDF 文件仅用于学习和演示目的，不得用于商业用途。

如果您是版权方并希望移除相关文件，请联系项目维护者。
