"""
ReAct 智能体工作流
基于 LlamaIndex 实现论文内容的交互式对话

完整的 ReAct (Reasoning and Acting) 范式实现：
1. 用户提问
2. Agent 思考 (Reasoning)
3. Agent 调用工具 (Acting)
4. 观察工具结果 (Observation)
5. 再次思考
6. 重复直到得出答案
"""
import yaml
from typing import List, Dict
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from graph_rag_tool import graph_rag_search, GraphRAGRetriever


class PaperQAAgent:
    """
    论文问答 ReAct 智能体
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_llm(self) -> OpenAI:
        """设置 LLM"""
        llm_config = self.config['llm']
        return OpenAI(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            api_base=llm_config['api_base'],
            temperature=0.7,
            max_tokens=3000
        )
    
    def _setup_tools(self) -> List[FunctionTool]:
        """
        设置智能体工具
        核心工具：Graph RAG 智能检索
        """
        tools = []
        
        # Tool 1: Graph RAG 检索工具
        graph_rag_tool = FunctionTool.from_defaults(
            fn=graph_rag_search,
            name="graph_rag_search",
            description=(
                "在知识图谱中进行智能检索和推理。"
                "该工具结合向量相似度检索和图结构分析，"
                "能够找到相关实体、它们的关系路径、所属知识社区及溯源信息。"
                "适用于需要深度理解和多跳推理的复杂问题。"
                "输入参数："
                "- query: 用户查询或关键词"
                "- top_k: 返回结果数量（默认5）"
                "- include_reasoning: 是否包含图推理（默认True）"
            )
        )
        tools.append(graph_rag_tool)
        
        # Tool 2: 实体详细信息查询
        def get_entity_details(entity_name: str) -> str:
            """
            获取指定实体的详细信息
            
            Args:
                entity_name: 实体名称
            
            Returns:
                实体的详细信息，包括属性、关系、社区等
            """
            retriever = GraphRAGRetriever()
            results = retriever.retrieve(
                entity_name, 
                top_k=1, 
                include_graph_reasoning=True
            )
            
            if not results['entities']:
                return f"未找到实体 '{entity_name}'"
            
            entity = results['entities'][0]
            output = [
                f"实体: {entity['name']}",
                f"层级: {entity['layer']}",
                f"社区ID: {entity['community_id']}",
            ]
            
            # 邻居关系
            if results['graph_reasoning'].get('neighbors'):
                neighbors = results['graph_reasoning']['neighbors']['neighbors']
                output.append(f"\n关联实体 ({len(neighbors)} 个):")
                for nb in neighbors[:5]:
                    output.append(f"  - {nb['name']} ({nb['distance']}跳)")
            
            # 溯源
            if results['source_documents']:
                docs = results['source_documents']
                output.append(f"\n知识来源: {docs[0]['document']}, 第{docs[0]['page']}页")
            
            return "\n".join(output)
        
        entity_tool = FunctionTool.from_defaults(
            fn=get_entity_details,
            name="get_entity_details",
            description=(
                "获取知识图谱中指定实体的详细信息。"
                "包括实体的层级、所属社区、关联实体和知识来源。"
                "适用于需要了解某个具体概念或实体的详细情况。"
            )
        )
        tools.append(entity_tool)
        
        # Tool 3: 关系路径查找
        def find_relationship_path(entity1: str, entity2: str) -> str:
            """
            查找两个实体之间的关系路径
            
            Args:
                entity1: 第一个实体名称
                entity2: 第二个实体名称
            
            Returns:
                两个实体之间的关系路径
            """
            retriever = GraphRAGRetriever()
            paths = retriever.find_paths_between_entities(entity1, entity2, max_length=4)
            
            if not paths:
                return f"未找到 '{entity1}' 和 '{entity2}' 之间的直接路径"
            
            output = [f"找到 {len(paths)} 条路径:\n"]
            for i, path in enumerate(paths[:3], 1):
                path_str = " -> ".join(path['nodes'])
                output.append(f"{i}. {path_str} (长度: {path['length']})")
            
            return "\n".join(output)
        
        path_tool = FunctionTool.from_defaults(
            fn=find_relationship_path,
            name="find_relationship_path",
            description=(
                "查找知识图谱中两个实体之间的关系路径。"
                "返回连接两个实体的最短路径，揭示概念之间的隐藏联系。"
                "适用于探索概念之间的关联关系。"
            )
        )
        tools.append(path_tool)
        
        return tools
    
    def _create_agent(self) -> ReActAgent:
        """
        创建 ReAct Agent
        """
        agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=15,  # 最多15轮思考-行动循环
            context=(
                "你是一个论文内容分析专家，擅长解答关于科研论文的问题。"
                "你可以使用知识图谱检索工具来获取论文中的相关信息。"
                "在回答问题时，请：\n"
                "1. 先思考问题需要哪些信息\n"
                "2. 使用工具检索相关知识\n"
                "3. 基于检索结果进行推理\n"
                "4. 给出准确、有依据的答案\n"
                "5. 提供知识溯源信息（来自哪篇论文、哪一页）\n\n"
                "请始终保持严谨和客观，如果检索结果中没有相关信息，请明确说明。"
            )
        )
        return agent
    
    def chat(self, user_query: str) -> str:
        """
        与用户对话
        
        Args:
            user_query: 用户提问
        
        Returns:
            Agent 的回答
        """
        print(f"\n{'='*70}")
        print(f"用户: {user_query}")
        print('='*70)
        print("\n🤖 Agent 思考过程:\n")
        
        response = self.agent.chat(user_query)
        
        print(f"\n{'='*70}")
        print("✅ 最终回答:")
        print('='*70)
        print(f"\n{response}\n")
        
        return str(response)
    
    def reset(self):
        """重置对话历史"""
        self.agent.reset()
        print("✓ 对话历史已清空")


def interactive_mode():
    """交互式对话模式"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ReAct 智能体 - 论文内容问答系统                      ║
║                                                              ║
║  基于 LlamaIndex ReAct Agent 实现                            ║
║  支持多步推理、工具调用、知识溯源                            ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("正在初始化 ReAct Agent...")
    try:
        agent = PaperQAAgent()
        print("✓ Agent 初始化完成\n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 70)
    print("交互式对话已启动")
    print("=" * 70)
    print("\n可用命令:")
    print("  - 直接输入问题进行对话")
    print("  - 输入 'reset' 清空对话历史")
    print("  - 输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            user_input = input("👤 您: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n再见！")
                break
            
            if user_input.lower() == 'reset':
                agent.reset()
                continue
            
            # 调用 Agent
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            print()


def demo_mode():
    """演示模式：预设问题"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ReAct 智能体 - 演示模式                             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    agent = PaperQAAgent()
    
    # 预设问题（由简到难）
    demo_questions = [
        "论文中提到了哪些关于融资的概念？",
        "Demo Day 和融资决策之间有什么关系？请详细说明。",
        "创业团队在融资过程中需要注意什么？请结合论文中的多个知识点回答。",
        "融资策略、Demo Day、创业者成功这三个概念之间存在什么样的关联？",
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#'*70}")
        print(f"# 演示问题 {i}/{len(demo_questions)}")
        print(f"{'#'*70}\n")
        
        agent.chat(question)
        
        if i < len(demo_questions):
            input("\n按 Enter 继续下一个问题...")
    
    print("\n\n演示完成！")


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
