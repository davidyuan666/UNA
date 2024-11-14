import pandas as pd
import os
import json
from python_bpe_encoder import PythonBPETokenizer
from transformers import BertTokenizer, BertModel
import torch

class PythonCodeEvaluator:
    def __init__(self, python_files_path, bpe_vocab_size=8192):
        """
        初始化评估器
        :param python_files_path: Python文件所在的目录路径
        """
        self.python_files_path = python_files_path
        self.evaluation_results = []
        self.python_bpe_encoder = PythonBPETokenizer(vocab_size=bpe_vocab_size)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def read_jsonl_file(self, file_path):
        """
        读取JSONL文件内容
        :param file_path: JSONL文件路径
        :return: Python代码列表
        """
        python_codes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if 'code' in data:
                        python_codes.append(data['code'])
        except Exception as e:
            print(f"读取文件出错: {e}")
        return python_codes

    def encode_with_bpe(self, code):
        """
        使用PythonBPETokenizer对代码进行编码
        :param code: Python代码
        :return: BPE编码后的代码
        """
        return self.python_bpe_encoder.transform(code, modality='code')

    def generate_summary(self, code):
        """
        使用BERT生成代码摘要
        :param code: Python代码
        :return: 代码摘要
        """
        inputs = self.bert_tokenizer(code, return_tensors='pt', truncation=True, max_length=512)
        outputs = self.bert_model(**inputs)
        # 简单地使用CLS token的输出作为摘要
        summary_vector = outputs.last_hidden_state[:, 0, :]
        return summary_vector

    def process_jsonl_files(self):
        """
        处理目录下的所有JSONL文件
        """
        for root, _, files in os.walk(self.python_files_path):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    python_codes = self.read_jsonl_file(file_path)
                    for code in python_codes:
                        bpe_encoded_code = self.encode_with_bpe(code)
                        bpe_summary = self.generate_summary(bpe_encoded_code)
                        non_bpe_summary = self.generate_summary(code)
                        # 人工评分需要手动输入或其他方式获取
                        human_score = 0  # 替换为实际的评分获取逻辑
                        self.add_evaluation(code, non_bpe_summary, bpe_summary, human_score)

    def add_evaluation(self, code_segment, non_bpe_summary, bpe_summary, human_score):
        """
        添加一条评估记录
        :param code_segment: 原始代码片段
        :param non_bpe_summary: 非BPE生成的摘要
        :param bpe_summary: BPE生成的摘要
        :param human_score: 人工评分
        """
        self.evaluation_results.append({
            'original_code': code_segment,
            'non_bpe_summary': non_bpe_summary,
            'bpe_summary': bpe_summary,
            'human_score': human_score
        })

    def export_to_csv(self, output_path):
        """
        将评估结果导出为CSV文件
        :param output_path: 输出CSV文件路径
        """
        df = pd.DataFrame(self.evaluation_results)
        df.to_csv(output_path, index=False, encoding='utf-8')