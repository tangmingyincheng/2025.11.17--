"""
社区摘要生成模块
为每个社区生成自然语言摘要，包括核心实体、主题和关系模式
"""
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from llama_index.llms.openai import OpenAI
from tqdm import tqdm


class CommunitySummaryGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password123')
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", neo4j_password)
        )
        self.output_dir = Path("community_summaries")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_llm(self) -> OpenAI:
        llm_config = self.config['llm']
        return OpenAI(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            api_base=llm_config['api_base'],
            temperature=0.3,
            max_tokens=1000
        )
    
    def get_communities(self) -> list:
        """获取所有社区信息"""
        query = """
        MATCH (e:Entity)
        WHERE e.community_id IS NOT NULL
        WITH e.community_id AS community_id, 
             COLLECT(DISTINCT e) AS entities
        RETURN community_id, entities
        ORDER BY community_id
        """
        with self.driver.session() as session:
            result = session.run(query)
            communities = []
            for record in result:
                communities.append({
                    'community_id': record['community_id'],
                    'entities': [dict(e) for e in record['entities']]
                })
            return communities
    
    def get_community_relationships(self, entity_names: list) -> list:
        """获取社区内的关系"""
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE e1.name IN $names AND e2.name IN $names
        RETURN e1.name AS source, 
               type(r) AS relation, 
               e2.name AS target,
               r.predicate AS predicate_cn
        """
        with self.driver.session() as session:
            result = session.run(query, names=entity_names)
            return [dict(record) for record in result]
    
    def get_source_documents(self, entity_names: list) -> list:
        """获取实体的源文档"""
        query = """
        MATCH (e:Entity)-[:FROM]->(d:Document)
        WHERE e.name IN $names
        RETURN DISTINCT d.title AS document
        """
        with self.driver.session() as session:
            result = session.run(query, names=entity_names)
            return [record['document'] for record in result]
    
    def generate_summary(self, community_data: dict) -> str:
        """为单个社区生成摘要"""
        community_id = community_data['community_id']
        entities = community_data['entities']
        entity_names = [e['name'] for e in entities]
        
        # 获取关系
        relationships = self.get_community_relationships(entity_names)
        
        # 获取源文档
        source_docs = self.get_source_documents(entity_names)
        
        # 构建提示词
        prompt = self._build_summary_prompt(
            community_id, entities, relationships, source_docs
        )
        
        # 调用LLM生成摘要
        response = self.llm.complete(prompt)
        return response.text.strip()
    
    def _build_summary_prompt(self, community_id: int, entities: list, 
                              relationships: list, source_docs: list) -> str:
        """构建摘要生成提示词"""
        # 按层级分组实体
        layers = {}
        for entity in entities:
            layer = entity.get('layer', 'Unknown')
            layers.setdefault(layer, []).append(entity['name'])
        
        # 格式化关系
        rel_text = "\n".join([
            f"- {r['source']} --[{r.get('predicate_cn', r['relation'])}]--> {r['target']}"
            for r in relationships[:10]  # 限制数量
        ])
        
        prompt = f"""你是一个知识图谱分析专家。请为以下知识社区生成一个简洁但全面的摘要。

**社区ID**: {community_id}
**社区规模**: {len(entities)} 个实体

**实体分布**:
{self._format_layer_distribution(layers)}

**关键关系**:
{rel_text if rel_text else "（该社区内实体之间暂无直接关系）"}

**来源文档**: {', '.join(source_docs) if source_docs else '未知'}

**要求**:
1. 用2-3句话概括这个社区的核心主题
2. 识别社区中最重要的实体和它们的作用
3. 总结实体之间的关系模式
4. 指出这个社区在整体知识图谱中可能扮演的角色

请直接输出摘要内容，不要包含标题或其他格式。
"""
        return prompt
    
    def _format_layer_distribution(self, layers: dict) -> str:
        """格式化层级分布"""
        lines = []
        for layer, entities in layers.items():
            lines.append(f"- {layer}: {len(entities)} 个 ({', '.join(entities[:3])}{'...' if len(entities) > 3 else ''})")
        return "\n".join(lines)
    
    def save_summaries(self, summaries: list) -> str:
        """保存摘要到文件"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"community_summaries_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        
        return str(output_file)
    
    def write_summaries_to_neo4j(self, summaries: list):
        """将摘要写入Neo4j作为Community节点"""
        query = """
        MERGE (c:Community {community_id: $community_id})
        SET c.summary = $summary,
            c.size = $size,
            c.updated_at = datetime()
        WITH c
        MATCH (e:Entity {community_id: $community_id})
        MERGE (e)-[:BELONGS_TO]->(c)
        """
        with self.driver.session() as session:
            for summary_data in summaries:
                session.run(query, 
                          community_id=summary_data['community_id'],
                          summary=summary_data['summary'],
                          size=summary_data['size'])
        print("✓ 社区摘要已写入Neo4j")
    
    def run(self):
        """主执行流程"""
        print("\n=== 社区摘要生成开始 ===")
        
        # 获取社区
        communities = self.get_communities()
        print(f"发现 {len(communities)} 个社区")
        
        # 生成摘要
        summaries = []
        for community_data in tqdm(communities, desc="生成摘要"):
            try:
                summary_text = self.generate_summary(community_data)
                summaries.append({
                    'community_id': community_data['community_id'],
                    'size': len(community_data['entities']),
                    'summary': summary_text,
                    'entities': [e['name'] for e in community_data['entities']],
                    'generated_at': datetime.utcnow().isoformat() + 'Z'
                })
            except Exception as e:
                print(f"\n警告: 社区 {community_data['community_id']} 摘要生成失败: {e}")
                summaries.append({
                    'community_id': community_data['community_id'],
                    'size': len(community_data['entities']),
                    'summary': f"[自动生成] 包含 {len(community_data['entities'])} 个实体的知识社区",
                    'entities': [e['name'] for e in community_data['entities']],
                    'generated_at': datetime.utcnow().isoformat() + 'Z',
                    'error': str(e)
                })
        
        # 保存到文件
        output_file = self.save_summaries(summaries)
        print(f"摘要已保存: {output_file}")
        
        # 写入Neo4j
        self.write_summaries_to_neo4j(summaries)
        
        print("=== 社区摘要生成完成 ===\n")
        return summaries
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()


if __name__ == "__main__":
    generator = CommunitySummaryGenerator()
    generator.run()
