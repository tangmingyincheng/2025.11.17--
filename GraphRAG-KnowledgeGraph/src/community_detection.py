#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Neo4j GDS进行社区发现
"""

import json
from datetime import datetime
from pathlib import Path

from graphdatascience import GraphDataScience
import yaml


class CommunityDetector:
    def __init__(self, config_path: str = "graphrag_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        neo4j_cfg = self.config["neo4j"]
        self.gds = GraphDataScience(neo4j_cfg["uri"],
                                     auth=(neo4j_cfg["username"], neo4j_cfg["password"]))
        self.database = neo4j_cfg.get("database", "neo4j")

        # 结果输出目录
        output_dir = Path(self.config["output"]["community_report_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = output_dir

    def run(self):
        print("\n=== 社区发现开始 ===")
        start_time = datetime.utcnow()

        # 创建投影
        graph_name = "kg_projection"
        self._drop_existing_graph(graph_name)
        graph_name = self._project_graph(graph_name)

        # 运行社区算法
        algorithm = self.config["community_detection"].get("algorithm", "louvain")
        if algorithm == "louvain":
            communities = self._run_louvain(graph_name)
        elif algorithm == "label_propagation":
            communities = self._run_label_propagation(graph_name)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        print(f"检测到 {len(communities)} 个社区")

        # 写回Neo4j
        self._write_back_communities(graph_name, communities)

        # 生成报告
        report_path = self._save_report(communities, start_time)
        print(f"社区报告已保存: {report_path}")

        # 清理
        self.gds.graph.drop(graph_name)
        print("=== 社区发现结束 ===")

    def _drop_existing_graph(self, graph_name: str):
        result = self.gds.graph.exists(graph_name)
        exists = result["exists"]
        if exists:
            print(f"删除已有投影: {graph_name}")
            self.gds.graph.drop(graph_name)

    def _project_graph(self, graph_name: str):
        print("创建图投影...")
        node_projection = "Entity"

        relationship_projection = {
            "RELATES": {
                "type": "RELATED_TO",
                "orientation": "UNDIRECTED",
            },
            "HELPS": {
                "type": "HELPS",
                "orientation": "UNDIRECTED",
            },
            "PROMOTES": {
                "type": "PROMOTES",
                "orientation": "UNDIRECTED",
            },
            "INFLUENCES": {
                "type": "INFLUENCES",
                "orientation": "UNDIRECTED",
            },
            "REQUIRES": {
                "type": "REQUIRES",
                "orientation": "UNDIRECTED",
            },
            "REGRETS": {
                "type": "REGRETS",
                "orientation": "UNDIRECTED",
            }
        }

        graph = self.gds.graph.project(graph_name,
                                       node_projection,
                                       relationship_projection)
        print(f"投影完成: 图名 {graph_name}")
        return graph_name

    def _run_louvain(self, graph_name):
        cfg = self.config["community_detection"].get("louvain", {})
        print("运行 Louvain 社区发现...")
        graph = self.gds.graph.get(graph_name)
        result = self.gds.louvain.stream(graph,
                                         maxLevels=cfg.get("max_levels", 10),
                                         maxIterations=cfg.get("max_iterations", 10))
        communities = {}
        for _, row in result.iterrows():
            node_id = row['nodeId']
            community_id = row['communityId']
            communities.setdefault(community_id, []).append(node_id)
        return communities

    def _run_label_propagation(self, graph_name):
        print("运行 Label Propagation 社区发现...")
        graph = self.gds.graph.get(graph_name)
        result = self.gds.labelPropagation.stream(graph)
        communities = {}
        for _, row in result.iterrows():
            node_id = row['nodeId']
            community_id = row['communityId']
            communities.setdefault(community_id, []).append(node_id)
        return communities

    def _write_back_communities(self, graph_name, communities):
        print("写回社区信息到Neo4j...")
        graph = self.gds.graph.get(graph_name)
        # 直接使用 mutate 模式
        self.gds.louvain.mutate(graph, mutateProperty="community_id")
        # 写回到 Neo4j
        self.gds.graph.nodeProperties.write(graph, ["community_id"])
        print("社区属性已写回 Neo4j")

    def _save_report(self, communities, start_time):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_dir / f"communities_{timestamp}.json"
        report_data = {
            "generated_at": start_time.isoformat() + "Z",
            "total_communities": len(communities),
            "communities": []
        }

        for cid, nodes in communities.items():
            report_data["communities"].append({
                "community_id": int(cid),
                "size": len(nodes),
                "nodes": [int(nid) for nid in nodes]
            })

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        return report_file


if __name__ == "__main__":
    detector = CommunityDetector()
    detector.run()
