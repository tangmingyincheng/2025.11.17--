"""
图谱向量化模块
为实体、关系和社区生成向量嵌入
"""
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


class GraphVectorizer:
    def __init__(self, config_path: str = "graphrag_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = self._load_embedding_model()
        neo4j_password = os.getenv('NEO4J_PASSWORD', 'password123')
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", neo4j_password)
        )
        self.output_dir = Path("embeddings")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_embedding_model(self):
        """加载嵌入模型"""
        model_name = self.config['vectorization']['embedding_model']
        print(f"加载嵌入模型: {model_name}")
        return SentenceTransformer(model_name)
    
    def get_entities(self) -> list:
        """获取所有实体"""
        query = """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[:FROM]->(d:Document)
        RETURN id(e) AS node_id,
               e.name AS name,
               e.layer AS layer,
               e.community_id AS community_id,
               COLLECT(DISTINCT d.title) AS source_docs
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def get_relationships(self) -> list:
        """获取所有语义关系"""
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE type(r) <> 'FROM' AND type(r) <> 'BELONGS_TO'
        RETURN id(r) AS rel_id,
               e1.name AS source,
               type(r) AS relation,
               e2.name AS target,
               r.predicate AS predicate_cn
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def get_communities(self) -> list:
        """获取社区摘要"""
        query = """
        MATCH (c:Community)
        RETURN c.community_id AS community_id,
               c.summary AS summary,
               c.size AS size
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def vectorize_entities(self, entities: list) -> dict:
        """向量化实体"""
        print("\n向量化实体...")
        texts = []
        metadata = []
        
        for entity in entities:
            # 构建富文本表示
            text_parts = [f"实体: {entity['name']}"]
            if entity['layer']:
                text_parts.append(f"层级: {entity['layer']}")
            if entity['source_docs']:
                text_parts.append(f"来源: {', '.join(entity['source_docs'])}")
            
            text = " | ".join(text_parts)
            texts.append(text)
            metadata.append({
                'node_id': entity['node_id'],
                'name': entity['name'],
                'layer': entity['layer'],
                'community_id': entity['community_id'],
                'type': 'entity'
            })
        
        # 批量生成嵌入
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        return {
            'embeddings': embeddings.tolist(),
            'metadata': metadata
        }
    
    def vectorize_relationships(self, relationships: list) -> dict:
        """向量化关系"""
        print("\n向量化关系...")
        texts = []
        metadata = []
        
        for rel in relationships:
            # 构建三元组文本
            predicate = rel.get('predicate_cn') or rel['relation']
            text = f"{rel['source']} {predicate} {rel['target']}"
            texts.append(text)
            metadata.append({
                'rel_id': rel['rel_id'],
                'source': rel['source'],
                'relation': rel['relation'],
                'target': rel['target'],
                'type': 'relationship'
            })
        
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        return {
            'embeddings': embeddings.tolist(),
            'metadata': metadata
        }
    
    def vectorize_communities(self, communities: list) -> dict:
        """向量化社区摘要"""
        print("\n向量化社区...")
        texts = []
        metadata = []
        
        for comm in communities:
            if comm['summary']:
                texts.append(comm['summary'])
                metadata.append({
                    'community_id': comm['community_id'],
                    'size': comm['size'],
                    'type': 'community'
                })
        
        if not texts:
            print("警告: 没有社区摘要可供向量化")
            return {'embeddings': [], 'metadata': []}
        
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        return {
            'embeddings': embeddings.tolist(),
            'metadata': metadata
        }
    
    def save_embeddings(self, all_embeddings: dict) -> str:
        """保存嵌入到文件"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"graph_embeddings_{timestamp}.json"
        
        # 转换numpy为list以便序列化
        data = {
            'model': self.config['vectorization']['embedding_model'],
            'dimension': 768,  # paraphrase-multilingual-mpnet-base-v2 dimension
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'entities': all_embeddings['entities'],
            'relationships': all_embeddings['relationships'],
            'communities': all_embeddings['communities']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(output_file)
    
    def run(self):
        """主执行流程"""
        print("\n=== 图谱向量化开始 ===")
        
        # 获取数据
        entities = self.get_entities()
        relationships = self.get_relationships()
        communities = self.get_communities()
        
        print(f"\n实体数量: {len(entities)}")
        print(f"关系数量: {len(relationships)}")
        print(f"社区数量: {len(communities)}")
        
        # 向量化
        all_embeddings = {
            'entities': self.vectorize_entities(entities),
            'relationships': self.vectorize_relationships(relationships),
            'communities': self.vectorize_communities(communities)
        }
        
        # 保存
        output_file = self.save_embeddings(all_embeddings)
        print(f"\n✓ 嵌入已保存: {output_file}")
        
        # 统计
        total_vectors = (
            len(all_embeddings['entities']['embeddings']) +
            len(all_embeddings['relationships']['embeddings']) +
            len(all_embeddings['communities']['embeddings'])
        )
        print(f"✓ 总计生成 {total_vectors} 个向量")
        print("=== 图谱向量化完成 ===\n")
        
        return output_file
    
    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.close()


if __name__ == "__main__":
    vectorizer = GraphVectorizer()
    vectorizer.run()
