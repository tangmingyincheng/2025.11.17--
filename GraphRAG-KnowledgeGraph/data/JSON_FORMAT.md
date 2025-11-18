# JSON 数据格式规范

## 1. 三元组输出格式 (triples_output.json)

### 结构说明
```json
{
  "metadata": {
    "total_triples": 整数,
    "source_files": ["源文件列表"],
    "extracted_at": "ISO时间戳",
    "llm_model": "模型名称"
  },
  "triples": [
    {
      "subject": "主语实体",
      "predicate": "谓词/关系",
      "object": "宾语实体",
      "confidence": 0.0-1.0,
      "source_file": "来源PDF文件名",
      "page_number": 页码,
      "block_id": 文本块ID,
      "source_text": "原始文本片段",
      "extracted_at": "提取时间戳"
    }
  ]
}
```

### 字段定义
- **metadata**: 元数据信息
  - `total_triples`: 提取的三元组总数
  - `source_files`: 所有源文件名列表
  - `extracted_at`: 提取时间（ISO 8601格式）
  - `llm_model`: 使用的LLM模型名称

- **triples**: 三元组数组
  - `subject`: 主语，通常是实体或概念
  - `predicate`: 谓词，描述主宾之间的关系
  - `object`: 宾语，通常是另一个实体或概念
  - `confidence`: 置信度分数 (0-1)
  - `source_file`: 该三元组来源的PDF文件
  - `page_number`: 所在页码
  - `block_id`: 文本块标识符
  - `source_text`: 提取该三元组的原始文本
  - `extracted_at`: 该条三元组的提取时间

### 示例
```json
{
  "metadata": {
    "total_triples": 21,
    "source_files": [
      "lesson6_1.json",
      "早期创业融资指南-奇绩创坛.json"
    ],
    "extracted_at": "2025-11-17T15:48:23.284579Z",
    "llm_model": "gpt-4o-mini"
  },
  "triples": [
    {
      "subject": "融资策略",
      "predicate": "帮助",
      "object": "团队创造契机",
      "confidence": 0.8,
      "source_file": "lesson6_1.pdf",
      "page_number": 1,
      "block_id": 2,
      "source_text": "这份指南的目的是提供具体融资策略，帮助团队创造契机",
      "extracted_at": "2025-11-17T15:44:24.644345Z"
    }
  ]
}
```

## 2. PDF 解析输出格式 (output_json/*.json)

### 结构说明
```json
{
  "file_name": "PDF文件名",
  "total_pages": 总页数,
  "extracted_at": "提取时间",
  "pages": [
    {
      "page_number": 页码,
      "blocks": [
        {
          "block_id": 块ID,
          "block_type": "类型",
          "text": "文本内容",
          "bbox": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

### 字段定义
- `file_name`: 原始PDF文件名
- `total_pages`: PDF总页数
- `extracted_at`: 解析时间戳
- `pages`: 页面数组
  - `page_number`: 页码（从1开始）
  - `blocks`: 文本块数组
    - `block_id`: 块的唯一标识符
    - `block_type`: 块类型（paragraph, title, list等）
    - `text`: 提取的文本内容
    - `bbox`: 边界框坐标 [左, 上, 右, 下]

## 3. Neo4j 导入格式

### 实体节点
```cypher
CREATE (e:Entity {
  name: "实体名称",
  layer: "层级",
  page_number: 页码,
  block_id: 块ID,
  source_file: "来源文件"
})
```

### 关系
```cypher
CREATE (e1)-[:关系类型 {
  predicate: "谓词",
  confidence: 置信度,
  source_text: "原始文本"
}]->(e2)
```

### 文档节点
```cypher
CREATE (d:Document {
  file_name: "文件名",
  title: "标题",
  total_pages: 总页数
})
```

## 4. 社区摘要格式

```json
{
  "community_id": 社区ID,
  "summary": "社区摘要",
  "key_entities": ["关键实体列表"],
  "insights": "洞察分析",
  "entity_count": 实体数量
}
```

## 使用说明

1. **提取流程**: PDF → parse_pdfs.py → JSON → extract_triples.py → triples_output.json
2. **导入流程**: triples_output.json → import_to_neo4j.py → Neo4j图数据库
3. **检索流程**: Neo4j + Qdrant → graph_rag_tool.py → 混合检索结果

所有JSON文件使用 UTF-8 编码，时间戳遵循 ISO 8601 标准。
