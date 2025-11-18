#!/usr/bin/env python3
"""
extract_triples.py

基于 LlamaIndex 框架的知识三元组抽取智能体工作流。
从 JSON 格式的 PDF 解析结果中抽取实体及其因果关系三元组。
"""
import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import yaml

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage


# 知识抽取提示词模板
KNOWLEDGE_EXTRACTION_PROMPT = """你是一位专业的知识图谱抽取专家。请从下面的文本中抽取实体和它们之间的因果关系三元组。

【任务要求】
1. 识别文本中的关键实体（人物、组织、概念、技术、产品等）
2. 抽取实体之间的因果关系、影响关系、属性关系
3. 每个三元组包含：主体(subject)、谓词(predicate)、客体(object)
4. 优先抽取因果关系，例如"导致"、"影响"、"促进"、"提升"、"具有"等

【输出格式】
请以 JSON 数组格式输出，每个三元组包含以下字段：
- subject: 主体实体
- predicate: 关系谓词
- object: 客体实体
- confidence: 置信度(0-1之间的浮点数)

如果文本中没有明显的实体关系，返回空数组 []

【待分析文本】
{text}

【输出示例】
[
  {{"subject": "人工智能", "predicate": "促进", "object": "生产效率提升", "confidence": 0.9}},
  {{"subject": "深度学习", "predicate": "依赖", "object": "大规模数据集", "confidence": 0.85}}
]

请仅输出 JSON 数组，不要包含其他说明文字："""


class KnowledgeExtractionWorkflow:
    """知识抽取智能体工作流"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化工作流"""
        self.config = self._load_config(config_path)
        self._setup_llm()
        self.extracted_triples = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # 默认配置
        return {
            'llm': {
                'provider': 'openai',  # 或 'openrouter'
                'model': 'gpt-4o-mini',
                'temperature': 0.1,
                'max_tokens': 2000,
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'api_base': os.getenv('OPENAI_API_BASE', None)
            },
            'extraction': {
                'sample_size': 2,  # 随机选择的 JSON 文件数量
                'max_blocks_per_file': 50,  # 每个文件最多处理的文本块数量
                'min_text_length': 20,  # 最小文本长度
                'batch_size': 1  # 批处理大小
            }
        }
    
    def _setup_llm(self):
        """配置 LLM"""
        llm_config = self.config['llm']
        
        # 支持 OpenAI 或 OpenRouter
        if llm_config['provider'] == 'openai':
            self.llm = OpenAI(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=llm_config['api_key'],
                api_base=llm_config.get('api_base')
            )
        else:
            # OpenRouter 配置
            self.llm = OpenAI(
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens'],
                api_key=llm_config['api_key'],
                api_base='https://openrouter.ai/api/v1'
            )
        
        Settings.llm = self.llm
    
    def select_random_samples(self, json_dir: str, sample_size: int) -> List[str]:
        """随机选择 JSON 文件作为样本"""
        json_files = list(Path(json_dir).glob('*.json'))
        
        if len(json_files) == 0:
            raise ValueError(f"未找到 JSON 文件: {json_dir}")
        
        # 如果文件数少于样本大小，使用全部文件
        actual_sample_size = min(sample_size, len(json_files))
        selected = random.sample(json_files, actual_sample_size)
        
        print(f"从 {len(json_files)} 个文件中随机选择了 {actual_sample_size} 个样本:")
        for f in selected:
            print(f"  - {f.name}")
        
        return [str(f) for f in selected]
    
    def extract_triples_from_text(self, text: str, source_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用 LLM 从文本中抽取三元组"""
        if len(text.strip()) < self.config['extraction']['min_text_length']:
            return []
        
        try:
            # 构建提示词
            prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(text=text)
            
            # 调用 LLM
            messages = [ChatMessage(role="user", content=prompt)]
            response = self.llm.chat(messages)
            
            # 解析响应
            response_text = response.message.content.strip()
            
            # 尝试提取 JSON 部分
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            triples = json.loads(response_text)
            
            # 添加溯源信息
            for triple in triples:
                triple.update({
                    'source_file': source_info['file_name'],
                    'page_number': source_info['page_number'],
                    'block_id': source_info['block_id'],
                    'source_text': text[:200] + '...' if len(text) > 200 else text,
                    'extracted_at': datetime.utcnow().isoformat() + 'Z'
                })
            
            return triples
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON 解析失败: {e}")
            print(f"  响应内容: {response_text[:200]}")
            return []
        except Exception as e:
            print(f"  ❌ 抽取失败: {e}")
            return []
    
    def process_json_file(self, json_path: str) -> List[Dict[str, Any]]:
        """处理单个 JSON 文件"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_name = data['file_name']
        triples_from_file = []
        
        max_blocks = self.config['extraction']['max_blocks_per_file']
        block_count = 0
        
        print(f"\n处理文件: {file_name}")
        
        for page in data['pages']:
            page_number = page['page_number']
            
            for block in page['blocks']:
                if block_count >= max_blocks:
                    print(f"  已达到最大处理块数限制 ({max_blocks})")
                    break
                
                if block['block_type'] != 'text' or not block.get('text'):
                    continue
                
                text = block['text'].strip()
                if not text:
                    continue
                
                block_count += 1
                
                source_info = {
                    'file_name': file_name,
                    'page_number': page_number,
                    'block_id': block['block_id']
                }
                
                triples = self.extract_triples_from_text(text, source_info)
                
                if triples:
                    print(f"  ✓ 页 {page_number} 块 {block['block_id']}: 抽取 {len(triples)} 个三元组")
                    triples_from_file.extend(triples)
            
            if block_count >= max_blocks:
                break
        
        print(f"文件 {file_name} 共抽取 {len(triples_from_file)} 个三元组")
        return triples_from_file
    
    def run(self, input_dir: str, output_dir: str):
        """执行知识抽取工作流"""
        print("=" * 60)
        print("知识三元组抽取智能体工作流")
        print("=" * 60)
        
        # 随机选择样本
        sample_size = self.config['extraction']['sample_size']
        selected_files = self.select_random_samples(input_dir, sample_size)
        
        # 处理每个文件
        all_triples = []
        for json_path in selected_files:
            triples = self.process_json_file(json_path)
            all_triples.extend(triples)
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'knowledge_triples_{timestamp}.json')
        
        output_data = {
            'metadata': {
                'total_triples': len(all_triples),
                'source_files': [os.path.basename(f) for f in selected_files],
                'extracted_at': datetime.utcnow().isoformat() + 'Z',
                'llm_model': self.config['llm']['model']
            },
            'triples': all_triples
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✓ 抽取完成！")
        print(f"  总三元组数: {len(all_triples)}")
        print(f"  输出文件: {output_file}")
        print("=" * 60)
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description="知识三元组抽取智能体工作流")
    parser.add_argument('--input-dir', '-i', required=True, help='输入 JSON 目录')
    parser.add_argument('--output-dir', '-o', default='knowledge_triples', help='输出目录')
    parser.add_argument('--config', '-c', help='配置文件路径 (YAML)')
    parser.add_argument('--sample-size', '-s', type=int, help='随机选择的文件数量')
    
    args = parser.parse_args()
    
    # 初始化工作流
    workflow = KnowledgeExtractionWorkflow(config_path=args.config)
    
    # 覆盖配置
    if args.sample_size:
        workflow.config['extraction']['sample_size'] = args.sample_size
    
    # 执行
    workflow.run(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
