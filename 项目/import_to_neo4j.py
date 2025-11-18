#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识图谱批量导入Neo4j脚本

功能:
1. 从三元组JSON文件读取知识三元组
2. 根据实体语义自动分配层级标签
3. 批量导入节点和关系到Neo4j
4. 添加完整的溯源元数据
5. 验证导入结果
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
import yaml
from tqdm import tqdm


class KnowledgeGraphImporter:
    """知识图谱导入器"""
    
    def __init__(self, config_path: str = "neo4j_config.yaml"):
        """初始化导入器"""
        self.config = self._load_config(config_path)
        self.driver = None
        self._connect_to_neo4j()
        
        # 层级分类关键词
        self.layer_keywords = {
            "MaterialLayer": ["材料", "物质", "化学", "元素", "原料", "成分"],
            "DeviceLayer": ["设备", "器件", "装置", "组件", "模块"],
            "SystemLayer": ["系统", "平台", "架构", "框架", "网络"],
            "ApplicationLayer": ["应用", "场景", "案例", "实践", "使用"],
            "ConceptLayer": ["概念", "理论", "策略", "方法", "思想", "原则", "融资", "投资", "决策"],
            "ProcessLayer": ["流程", "过程", "步骤", "阶段", "程序", "路演", "Demo Day"]
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _connect_to_neo4j(self):
        """连接到Neo4j数据库"""
        neo4j_config = self.config['neo4j']
        try:
            self.driver = GraphDatabase.driver(
                neo4j_config['uri'],
                auth=(neo4j_config['username'], neo4j_config['password'])
            )
            # 测试连接
            with self.driver.session(database=neo4j_config.get('database', 'neo4j')) as session:
                session.run("RETURN 1")
            print(f"✓ 成功连接到Neo4j: {neo4j_config['uri']}")
        except Exception as e:
            raise ConnectionError(f"无法连接到Neo4j数据库: {e}")
    
    def _infer_layer(self, entity: str) -> str:
        """根据实体名称推断所属层级"""
        for layer, keywords in self.layer_keywords.items():
            for keyword in keywords:
                if keyword in entity:
                    return layer
        # 默认归为概念层
        return "ConceptLayer"
    
    def _normalize_relation_type(self, predicate: str) -> str:
        """标准化关系类型"""
        relation_mapping = {
            "帮助": "HELPS",
            "促进": "PROMOTES",
            "影响": "INFLUENCES",
            "后悔": "REGRETS",
            "组成": "CONSISTS_OF",
            "相关": "RELATED_TO",
            "需要": "REQUIRES",
            "产生": "PRODUCES",
            "阻止": "PREVENTS",
            "包含": "CONTAINS",
            "属于": "BELONGS_TO"
        }
        return relation_mapping.get(predicate, "RELATED_TO")
    
    def clear_database(self):
        """清空数据库(慎用!)"""
        print("\n⚠️  警告: 即将清空Neo4j数据库中的所有数据!")
        response = input("确认要继续吗? (yes/no): ")
        if response.lower() != 'yes':
            print("已取消清空操作")
            return
        
        with self.driver.session(database=self.config['neo4j'].get('database', 'neo4j')) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ 数据库已清空")
    
    def create_indexes(self):
        """创建索引以提高查询性能"""
        print("\n创建索引...")
        with self.driver.session(database=self.config['neo4j'].get('database', 'neo4j')) as session:
            # 为不同层级的节点创建索引
            for layer_config in self.config['schema']['node_layers']:
                layer = layer_config['name']
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{layer}) ON (n.name)")
                except Exception as e:
                    print(f"  索引创建失败 ({layer}): {e}")
            
            # 为Entity基础标签创建索引
            try:
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)")
                print("✓ 索引创建完成")
            except Exception as e:
                print(f"  索引创建失败: {e}")
    
    def load_triples_from_json(self, json_path: str) -> List[Dict]:
        """从JSON文件加载三元组"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('triples', [])
    
    def import_triples_batch(self, triples: List[Dict], batch_size: int = 100):
        """批量导入三元组"""
        print(f"\n开始导入 {len(triples)} 个三元组...")
        
        with self.driver.session(database=self.config['neo4j'].get('database', 'neo4j')) as session:
            # 分批处理
            for i in tqdm(range(0, len(triples), batch_size), desc="导入进度"):
                batch = triples[i:i+batch_size]
                self._import_batch(session, batch)
        
        print("✓ 三元组导入完成")
    
    def _import_batch(self, session, batch: List[Dict]):
        """导入单个批次"""
        for triple in batch:
            try:
                # 提取三元组信息
                subject = triple.get('subject', '').strip()
                predicate = triple.get('predicate', '').strip()
                obj = triple.get('object', '').strip()
                
                if not all([subject, predicate, obj]):
                    continue
                
                # 推断层级
                subject_layer = self._infer_layer(subject)
                object_layer = self._infer_layer(obj)
                
                # 标准化关系类型
                relation_type = self._normalize_relation_type(predicate)
                
                # 准备元数据
                metadata = {
                    'source_file': triple.get('source_file', ''),
                    'page_number': triple.get('page_number', 0),
                    'block_id': triple.get('block_id', 0),
                    'confidence': triple.get('confidence', 0.0),
                    'source_text': triple.get('source_text', ''),
                    'extracted_at': triple.get('extracted_at', '')
                }
                
                # 创建主语节点
                session.run(
                    f"""
                    MERGE (s:Entity:{subject_layer} {{name: $subject}})
                    SET s.layer = $subject_layer,
                        s.source_file = $source_file,
                        s.page_number = $page_number,
                        s.block_id = $block_id,
                        s.extracted_at = $extracted_at
                    """,
                    subject=subject,
                    subject_layer=subject_layer,
                    **metadata
                )
                
                # 创建宾语节点
                session.run(
                    f"""
                    MERGE (o:Entity:{object_layer} {{name: $object}})
                    SET o.layer = $object_layer,
                        o.source_file = $source_file,
                        o.page_number = $page_number,
                        o.block_id = $block_id,
                        o.extracted_at = $extracted_at
                    """,
                    object=obj,
                    object_layer=object_layer,
                    **metadata
                )
                
                # 创建关系
                session.run(
                    f"""
                    MATCH (s:Entity {{name: $subject}})
                    MATCH (o:Entity {{name: $object}})
                    MERGE (s)-[r:{relation_type}]->(o)
                    SET r.predicate = $predicate,
                        r.confidence = $confidence,
                        r.source_text = $source_text,
                        r.source_file = $source_file,
                        r.page_number = $page_number,
                        r.block_id = $block_id
                    """,
                    subject=subject,
                    object=obj,
                    predicate=predicate,
                    **metadata
                )
                
            except Exception as e:
                print(f"\n导入三元组失败: {triple}")
                print(f"错误: {e}")
    
    def create_source_documents(self, triples: List[Dict]):
        """创建源文档节点并建立FROM关系"""
        print("\n创建源文档溯源关系...")
        
        # 提取所有唯一的源文档
        source_docs = set()
        for triple in triples:
            source_file = triple.get('source_file', '')
            if source_file:
                source_docs.add(source_file)
        
        with self.driver.session(database=self.config['neo4j'].get('database', 'neo4j')) as session:
            # 创建文档节点
            for doc in source_docs:
                session.run(
                    """
                    MERGE (d:Document {name: $doc})
                    SET d.type = 'source_document'
                    """,
                    doc=doc
                )
            
            # 建立FROM关系
            session.run(
                """
                MATCH (e:Entity)
                WHERE e.source_file IS NOT NULL
                MATCH (d:Document {name: e.source_file})
                MERGE (e)-[r:FROM]->(d)
                """
            )
        
        print(f"✓ 已创建 {len(source_docs)} 个源文档节点及溯源关系")
    
    def validate_graph(self):
        """验证图谱结构"""
        print("\n" + "="*60)
        print("图谱验证结果")
        print("="*60)
        
        with self.driver.session(database=self.config['neo4j'].get('database', 'neo4j')) as session:
            # 执行验证查询
            for validation in self.config['validation']['queries']:
                print(f"\n【{validation['name']}】")
                result = session.run(validation['query'])
                for record in result:
                    print(f"  {dict(record)}")
        
        print("\n" + "="*60)
    
    def get_latest_triples_file(self) -> str:
        """获取最新的三元组文件"""
        triples_dir = Path(self.config['import']['triples_dir'])
        if not triples_dir.exists():
            raise FileNotFoundError(f"三元组目录不存在: {triples_dir}")
        
        json_files = list(triples_dir.glob("knowledge_triples_*.json"))
        if not json_files:
            raise FileNotFoundError(f"未找到三元组文件: {triples_dir}")
        
        # 按文件修改时间排序,选择最新的
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def run_import(self, clear_db: bool = False):
        """执行完整的导入流程"""
        print("\n" + "="*60)
        print("知识图谱导入器")
        print("="*60)
        
        # 清空数据库(可选)
        if clear_db:
            self.clear_database()
        
        # 创建索引
        self.create_indexes()
        
        # 获取三元组文件
        triples_file = self.config['import'].get('specific_file')
        if not triples_file or self.config['import'].get('use_latest', True):
            triples_file = self.get_latest_triples_file()
        
        print(f"\n使用三元组文件: {triples_file}")
        
        # 加载三元组
        triples = self.load_triples_from_json(triples_file)
        print(f"加载了 {len(triples)} 个三元组")
        
        # 批量导入
        batch_size = self.config['import'].get('batch_size', 100)
        self.import_triples_batch(triples, batch_size)
        
        # 创建源文档溯源
        self.create_source_documents(triples)
        
        # 验证
        if self.config['validation'].get('enabled', True):
            self.validate_graph()
        
        print("\n✓ 导入完成!")
        print("\n提示: 现在可以在Neo4j Browser中查看图谱")
        print(f"  访问: http://localhost:7474")
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将知识三元组导入Neo4j')
    parser.add_argument('-c', '--config', default='neo4j_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--clear', action='store_true',
                        help='导入前清空数据库')
    parser.add_argument('-f', '--file', 
                        help='指定三元组JSON文件(默认使用最新的)')
    
    args = parser.parse_args()
    
    try:
        # 创建导入器
        importer = KnowledgeGraphImporter(args.config)
        
        # 如果指定了文件,更新配置
        if args.file:
            importer.config['import']['specific_file'] = args.file
            importer.config['import']['use_latest'] = False
        
        # 执行导入
        importer.run_import(clear_db=args.clear)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'importer' in locals():
            importer.close()


if __name__ == "__main__":
    main()
