"""
Qdrant向量数据库集成模块
将图谱嵌入导入Qdrant
"""
import json
import yaml
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm


class QdrantSetup:
    def __init__(self, config_path: str = "graphrag_config.yaml"):
        self.config = self._load_config(config_path)
        self.client = QdrantClient(
            host=self.config['vector_store']['qdrant']['host'],
            port=self.config['vector_store']['qdrant']['port']
        )
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_collections(self):
        """创建Qdrant集合"""
        dimension = 768  # paraphrase-multilingual-mpnet-base-v2 dimension
        
        collections = [
            ('kg_entities', '知识图谱实体'),
            ('kg_relationships', '知识图谱关系'),
            ('kg_communities', '知识图谱社区')
        ]
        
        for collection_name, description in collections:
            try:
                # 删除已存在的集合
                self.client.delete_collection(collection_name)
                print(f"已删除旧集合: {collection_name}")
            except:
                pass
            
            # 创建新集合
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ 创建集合: {collection_name} ({description})")
    
    def load_latest_embeddings(self) -> dict:
        """加载最新的嵌入文件"""
        embeddings_dir = Path("embeddings")
        files = sorted(embeddings_dir.glob("graph_embeddings_*.json"))
        
        if not files:
            raise FileNotFoundError("未找到嵌入文件，请先运行 vectorize_graph.py")
        
        latest_file = files[-1]
        print(f"加载嵌入文件: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def upload_entities(self, data: dict):
        """上传实体向量"""
        print("\n上传实体向量...")
        points = []
        
        embeddings = data['entities']['embeddings']
        metadata = data['entities']['metadata']
        
        for idx, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            points.append(PointStruct(
                id=idx,
                vector=embedding,
                payload=meta
            ))
        
        # 批量上传
        self.client.upload_points(
            collection_name='kg_entities',
            points=points,
            batch_size=100
        )
        print(f"✓ 已上传 {len(points)} 个实体向量")
    
    def upload_relationships(self, data: dict):
        """上传关系向量"""
        print("\n上传关系向量...")
        points = []
        
        embeddings = data['relationships']['embeddings']
        metadata = data['relationships']['metadata']
        
        for idx, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            points.append(PointStruct(
                id=idx,
                vector=embedding,
                payload=meta
            ))
        
        self.client.upload_points(
            collection_name='kg_relationships',
            points=points,
            batch_size=100
        )
        print(f"✓ 已上传 {len(points)} 个关系向量")
    
    def upload_communities(self, data: dict):
        """上传社区向量"""
        print("\n上传社区向量...")
        
        embeddings = data['communities']['embeddings']
        metadata = data['communities']['metadata']
        
        if not embeddings:
            print("警告: 没有社区向量可上传")
            return
        
        points = []
        for idx, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            points.append(PointStruct(
                id=idx,
                vector=embedding,
                payload=meta
            ))
        
        self.client.upload_points(
            collection_name='kg_communities',
            points=points,
            batch_size=100
        )
        print(f"✓ 已上传 {len(points)} 个社区向量")
    
    def verify_setup(self):
        """验证导入"""
        print("\n=== Qdrant 集合统计 ===")
        for collection in ['kg_entities', 'kg_relationships', 'kg_communities']:
            info = self.client.get_collection(collection)
            print(f"{collection}: {info.points_count} 个向量")
    
    def run(self):
        """主执行流程"""
        print("\n=== Qdrant 向量数据库集成开始 ===")
        
        # 创建集合
        self.create_collections()
        
        # 加载嵌入
        embeddings_data = self.load_latest_embeddings()
        
        # 上传向量
        self.upload_entities(embeddings_data)
        self.upload_relationships(embeddings_data)
        self.upload_communities(embeddings_data)
        
        # 验证
        self.verify_setup()
        
        print("\n=== Qdrant 向量数据库集成完成 ===\n")


if __name__ == "__main__":
    setup = QdrantSetup()
    setup.run()
