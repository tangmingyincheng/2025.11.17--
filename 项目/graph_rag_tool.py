"""
Graph RAG 智能检索工具 (LlamaIndex Tool)
结合向量检索、图路径搜索和社区推理的混合检索系统
"""
import json
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field


class GraphRAGRetriever:
    """Graph RAG 检索器核心类"""
    
    def __init__(self, config_path: str = "graphrag_config.yaml"):
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.embedding_model = SentenceTransformer(
            self.config['vectorization']['embedding_model']
        )
        self.qdrant_client = QdrantClient(
            host=self.config['vector_store']['qdrant']['host'],
            port=self.config['vector_store']['qdrant']['port']
        )
        self.neo4j_driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123")
        )
        
        # 检索配置
        self.retrieval_config = self.config['retrieval']
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def vector_search_entities(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量检索实体"""
        query_vector = self.embedding_model.encode(query).tolist()
        
        results = self.qdrant_client.query_points(
            collection_name='kg_entities',
            query=query_vector,
            limit=top_k
        ).points
        
        return [{
            'name': hit.payload['name'],
            'layer': hit.payload['layer'],
            'community_id': hit.payload['community_id'],
            'node_id': hit.payload['node_id'],
            'score': hit.score,
            'type': 'entity'
        } for hit in results]
    
    def vector_search_communities(self, query: str, top_k: int = 3) -> List[Dict]:
        """向量检索社区"""
        query_vector = self.embedding_model.encode(query).tolist()
        
        try:
            results = self.qdrant_client.query_points(
                collection_name='kg_communities',
                query=query_vector,
                limit=top_k
            ).points
            
            return [{
                'community_id': hit.payload['community_id'],
                'size': hit.payload['size'],
                'score': hit.score,
                'type': 'community'
            } for hit in results]
        except Exception as e:
            print(f"警告: 社区向量检索失败: {e}")
            return []
    
    def get_community_summary(self, community_id: int) -> Optional[str]:
        """获取社区摘要"""
        query = """
        MATCH (c:Community {community_id: $community_id})
        RETURN c.summary AS summary
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, community_id=community_id)
            record = result.single()
            return record['summary'] if record else None
    
    def get_entity_neighbors(self, entity_name: str, max_hops: int = 2) -> Dict:
        """获取实体的多跳邻居"""
        query = """
        MATCH path = (e:Entity {name: $name})-[*1..%d]-(neighbor:Entity)
        WHERE e <> neighbor
        WITH e, neighbor, 
             [r IN relationships(path) | {type: type(r), predicate: r.predicate}] AS rels,
             length(path) AS distance
        RETURN DISTINCT 
               neighbor.name AS neighbor_name,
               neighbor.layer AS neighbor_layer,
               neighbor.community_id AS neighbor_community,
               rels,
               distance
        ORDER BY distance, neighbor_name
        LIMIT 20
        """ % max_hops
        
        with self.neo4j_driver.session() as session:
            result = session.run(query, name=entity_name)
            neighbors = []
            for record in result:
                neighbors.append({
                    'name': record['neighbor_name'],
                    'layer': record['neighbor_layer'],
                    'community_id': record['neighbor_community'],
                    'relationships': record['rels'],
                    'distance': record['distance']
                })
            return {'entity': entity_name, 'neighbors': neighbors}
    
    def find_paths_between_entities(self, entity1: str, entity2: str, 
                                    max_length: int = 3) -> List[Dict]:
        """查找两个实体之间的路径"""
        query = """
        MATCH path = shortestPath((e1:Entity {name: $name1})-[*..%d]-(e2:Entity {name: $name2}))
        WHERE e1 <> e2
        WITH path, length(path) AS path_length
        RETURN [n IN nodes(path) | n.name] AS nodes,
               [r IN relationships(path) | {type: type(r), predicate: r.predicate}] AS relationships,
               path_length
        ORDER BY path_length
        LIMIT 5
        """ % max_length
        
        with self.neo4j_driver.session() as session:
            result = session.run(query, name1=entity1, name2=entity2)
            paths = []
            for record in result:
                paths.append({
                    'nodes': record['nodes'],
                    'relationships': record['relationships'],
                    'length': record['path_length']
                })
            return paths
    
    def get_entity_source_documents(self, entity_name: str) -> List[str]:
        """获取实体的溯源文档"""
        query = """
        MATCH (e:Entity {name: $name})-[:FROM]->(d:Document)
        RETURN DISTINCT d.title AS document,
                        e.page_number AS page,
                        e.block_id AS block
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, name=entity_name)
            return [{
                'document': record['document'],
                'page': record['page'],
                'block': record['block']
            } for record in result]
    
    def retrieve(self, query: str, top_k: int = None, 
                 include_graph_reasoning: bool = True) -> Dict[str, Any]:
        """
        混合检索主函数
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            include_graph_reasoning: 是否包含图推理
        
        Returns:
            结构化的检索结果
        """
        if top_k is None:
            top_k = self.retrieval_config['top_k']
        
        results = {
            'query': query,
            'entities': [],
            'communities': [],
            'graph_reasoning': {},
            'source_documents': []
        }
        
        # 1. 向量检索实体
        entity_results = self.vector_search_entities(query, top_k)
        results['entities'] = entity_results
        
        # 2. 向量检索社区
        community_results = self.vector_search_communities(query, top_k=3)
        for comm in community_results:
            summary = self.get_community_summary(comm['community_id'])
            comm['summary'] = summary
        results['communities'] = community_results
        
        # 3. 图推理
        if include_graph_reasoning and entity_results:
            # 获取top实体的邻居
            top_entity = entity_results[0]['name']
            neighbors = self.get_entity_neighbors(top_entity, max_hops=2)
            results['graph_reasoning']['neighbors'] = neighbors
            
            # 如果有多个高分实体，查找它们之间的路径
            if len(entity_results) >= 2:
                entity1 = entity_results[0]['name']
                entity2 = entity_results[1]['name']
                paths = self.find_paths_between_entities(entity1, entity2)
                results['graph_reasoning']['paths'] = {
                    'from': entity1,
                    'to': entity2,
                    'paths': paths
                }
        
        # 4. 溯源文档
        seen_docs = set()
        for entity in entity_results[:3]:  # 取top3实体的溯源
            docs = self.get_entity_source_documents(entity['name'])
            for doc in docs:
                doc_key = f"{doc['document']}_{doc['page']}"
                if doc_key not in seen_docs:
                    results['source_documents'].append(doc)
                    seen_docs.add(doc_key)
        
        return results
    
    def format_results_for_llm(self, results: Dict[str, Any]) -> str:
        """格式化检索结果供LLM使用"""
        output = []
        
        # 社区摘要（优先级最高）
        if results['communities']:
            output.append("【相关知识社区】")
            for comm in results['communities']:
                output.append(f"社区 {comm['community_id']} (相似度: {comm['score']:.3f}):")
                output.append(f"  {comm['summary']}")
                output.append("")
        
        # 核心实体
        if results['entities']:
            output.append("【核心实体】")
            for ent in results['entities'][:5]:
                output.append(f"- {ent['name']} [{ent['layer']}] (相似度: {ent['score']:.3f})")
            output.append("")
        
        # 图推理
        if results['graph_reasoning'].get('neighbors'):
            neighbors_data = results['graph_reasoning']['neighbors']
            output.append(f"【{neighbors_data['entity']} 的关联实体】")
            for nb in neighbors_data['neighbors'][:5]:
                rel_desc = " -> ".join([r.get('predicate') or r.get('type', '未知') for r in nb['relationships']])
                output.append(f"- {nb['name']} ({nb['distance']}跳, 关系: {rel_desc})")
            output.append("")
        
        # 路径推理
        if results['graph_reasoning'].get('paths'):
            path_data = results['graph_reasoning']['paths']
            if path_data['paths']:
                output.append(f"【{path_data['from']} 到 {path_data['to']} 的路径】")
                for i, path in enumerate(path_data['paths'][:2], 1):
                    path_str = " -> ".join(path['nodes'])
                    output.append(f"{i}. {path_str}")
                output.append("")
        
        # 溯源信息
        if results['source_documents']:
            output.append("【知识溯源】")
            for doc in results['source_documents'][:3]:
                output.append(f"- {doc['document']}, 第{doc['page']}页")
            output.append("")
        
        return "\n".join(output)
    
    def __del__(self):
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()


# ============================================================================
# LlamaIndex Tool 封装
# ============================================================================

# 全局检索器实例（避免重复初始化）
_retriever_instance = None

def get_retriever() -> GraphRAGRetriever:
    """获取检索器单例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = GraphRAGRetriever()
    return _retriever_instance


def graph_rag_search(
    query: str,
    top_k: int = 5,
    include_reasoning: bool = True
) -> str:
    """
    Graph RAG 智能检索工具
    
    在知识图谱中进行混合检索，结合向量相似度和图结构推理。
    适用于需要深度知识理解和关系推理的复杂查询。
    
    Args:
        query: 用户查询或关键词
        top_k: 返回结果数量 (默认5)
        include_reasoning: 是否包含图推理 (默认True)
    
    Returns:
        格式化的检索结果，包含相关实体、社区摘要、关系路径和溯源信息
    """
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k, include_reasoning)
    return retriever.format_results_for_llm(results)


# 创建 LlamaIndex FunctionTool
graph_rag_tool = FunctionTool.from_defaults(
    fn=graph_rag_search,
    name="graph_rag_search",
    description=(
        "在知识图谱中进行智能检索和推理。"
        "该工具结合向量相似度检索和图结构分析，"
        "能够找到相关实体、它们的关系路径、所属知识社区及溯源信息。"
        "适用于需要深度理解和多跳推理的复杂问题。"
    )
)


# ============================================================================
# 测试和示例
# ============================================================================

if __name__ == "__main__":
    # 测试检索器
    retriever = GraphRAGRetriever()
    
    test_queries = [
        "融资策略",
        "Demo Day",
        "创业团队"
    ]
    
    print("=== Graph RAG 检索测试 ===\n")
    for query in test_queries:
        print(f"\n查询: {query}")
        print("=" * 60)
        results = retriever.retrieve(query, top_k=3, include_graph_reasoning=True)
        formatted = retriever.format_results_for_llm(results)
        print(formatted)
        print()
