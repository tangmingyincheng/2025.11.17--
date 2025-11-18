# Graph RAG 知识图谱智能检索系统

基于 Neo4j + Qdrant + LlamaIndex 构建的端到端 Graph RAG 系统，支持 PDF 解析、知识抽取、图谱构建、社区发现、向量检索和 ReAct 智能体对话。

## 📋 项目概述

本项目实现了完整的 Graph RAG（图增强检索生成）工作流：

1. **PDF 解析** → 结构化文本提取
2. **三元组抽取** → LLM 驱动的知识抽取
3. **图谱构建** → Neo4j 知识图谱建模
4. **社区发现** → Louvain 算法聚类
5. **向量化** → 实体/关系/社区嵌入
6. **混合检索** → 向量相似度 + 图推理
7. **ReAct Agent** → 智能对话问答

## 🏗️ 项目结构

```
GraphRAG-KnowledgeGraph/
├── src/                          # 源代码
│   ├── parse_pdfs.py            # PDF解析
│   ├── extract_triples.py       # 三元组抽取
│   ├── import_to_neo4j.py       # Neo4j导入
│   ├── community_detection.py   # 社区发现
│   ├── generate_community_summaries.py  # 社区摘要
│   ├── vectorize_graph.py       # 图向量化
│   ├── setup_qdrant.py          # Qdrant配置
│   ├── graph_rag_tool.py        # Graph RAG检索
│   └── react_agent.py           # ReAct智能体
├── data/                         # 数据文件
│   ├── pdfs/                    # 源PDF文件
│   ├── outputs/                 # 输出文件
│   │   └── triples_output.json  # 三元组结果
│   └── JSON_FORMAT.md           # JSON格式规范
├── configs/                      # 配置文件
│   └── config.yaml              # 主配置
├── docs/                         # 文档
│   └── 实践报告.md              # 项目报告
├── docker-compose.yml            # Docker配置
├── requirements.txt              # Python依赖
└── README.md                     # 本文件
```

## 📦 数据集下载

**PDF 源文件**（约 3.4 MB）：

由于 GitHub 不建议上传大型数据文件，请从网盘下载演示 PDF：

- 📥 **夸克网盘**: [11.17示例PDF](https://pan.quark.cn/s/c0fd289b3cae)

下载后放置到 `data/pdfs/` 目录。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- 4GB+ RAM

### 1. 安装依赖

```bash
# 创建虚拟环境
conda create -n graphrag python=3.10
conda activate graphrag

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动 Neo4j 和 Qdrant

```bash
# 启动 Docker 服务
docker-compose up -d

# 验证服务
# Neo4j Browser: http://localhost:7474 (neo4j/your_password)
# Qdrant: http://localhost:6333/dashboard
```

### 3. 配置 API 密钥

编辑 `configs/config.yaml`：

```yaml
llm:
  api_key: "your-api-key"
  api_base: "https://yunwu.ai/v1"
  model: "gpt-4o-mini"
```

### 4. 运行完整流程

```bash
cd src

# 步骤 1: 解析 PDF
python parse_pdfs.py

# 步骤 2: 抽取三元组
python extract_triples.py

# 步骤 3: 导入 Neo4j
python import_to_neo4j.py

# 步骤 4: 社区发现
python community_detection.py

# 步骤 5: 生成社区摘要
python generate_community_summaries.py

# 步骤 6: 向量化图谱
python vectorize_graph.py

# 步骤 7: 配置 Qdrant
python setup_qdrant.py

# 步骤 8: 启动 ReAct Agent
python react_agent.py          # 交互模式
python react_agent.py --demo   # 演示模式
```

## 💡 核心功能

### 1. PDF 解析

```python
from parse_pdfs import PDFParser

parser = PDFParser()
parser.parse_directory("data/pdfs", "data/outputs")
```

### 2. 知识抽取

```python
from extract_triples import TripleExtractor

extractor = TripleExtractor(config_path="configs/config.yaml")
triples = extractor.extract_from_directory("data/outputs")
```

### 3. Graph RAG 检索

```python
from graph_rag_tool import GraphRAGRetriever

retriever = GraphRAGRetriever()
results = retriever.retrieve(
    query="融资策略有哪些要点？",
    top_k=5,
    include_graph_reasoning=True
)
```

### 4. ReAct Agent 对话

```python
from react_agent import PaperQAAgent

agent = PaperQAAgent()
response = agent.chat("创业团队在融资过程中需要注意什么？")
```

## 📊 数据格式

详见 [`data/JSON_FORMAT.md`](data/JSON_FORMAT.md)

### 三元组格式示例

```json
{
  "subject": "融资策略",
  "predicate": "帮助",
  "object": "团队创造契机",
  "confidence": 0.8,
  "source_file": "lesson6_1.pdf",
  "page_number": 1
}
```

## 🔧 技术栈

- **LLM**: OpenAI API (gpt-4o-mini)
- **图数据库**: Neo4j 5.18.0 + GDS 2.6.7
- **向量数据库**: Qdrant latest
- **嵌入模型**: paraphrase-multilingual-mpnet-base-v2
- **Agent 框架**: LlamaIndex ReActAgent
- **Python**: 3.10+

## 📈 项目亮点

1. **端到端工作流** - 从 PDF 到对话的完整流程
2. **混合检索** - 向量相似度 + 图路径推理
3. **知识溯源** - 每个答案都标注来源页码
4. **多跳推理** - 支持复杂的图谱遍历
5. **社区摘要** - LLM 生成的语义聚类
6. **ReAct 范式** - 可解释的推理链

## 🎯 使用场景

- 学术论文问答
- 企业知识库检索
- 复杂文档分析
- 多跳关系推理
- 知识发现

## 📝 配置说明

### Neo4j 配置

```yaml
# docker-compose.yml
neo4j:
  image: neo4j:5.18.0-enterprise
  environment:
    - NEO4J_AUTH=neo4j/your_password_here
  ports:
    - "7474:7474"
    - "7687:7687"
```

### Qdrant 配置

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
```

## 🐛 常见问题

### 1. Neo4j GDS 版本不兼容

确保使用 Neo4j 5.18.0 + GDS 2.6.7 组合。

### 2. Document.title 缺失警告

运行以下 Cypher 修复：
```cypher
MATCH (d:Document)
WHERE d.title IS NULL
SET d.title = d.file_name
```

### 3. Qdrant 连接失败

检查 Docker 容器状态：
```bash
docker-compose ps
docker-compose logs qdrant
```

## 📖 详细文档

- [实践报告](docs/实践报告.md) - 完整的项目实践过程
- [JSON 格式规范](data/JSON_FORMAT.md) - 数据格式说明

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 GitHub Issue。

---

**注意**: 本项目使用的 PDF 文件仅用于演示，请勿用于商业用途。
