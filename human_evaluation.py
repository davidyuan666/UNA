import pandas as pd
import os

class JavaCodeEvaluator:
    def __init__(self, java_files_path):
        """
        初始化评估器
        :param java_files_path: Java文件所在的目录路径
        """
        self.java_files_path = java_files_path
        self.evaluation_results = []

    def read_java_file(self, file_path):
        """
        读取Java文件内容
        :param file_path: Java文件路径
        :return: 文件内容字符串
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"读取文件出错: {e}")
            return None

    def extract_code_segments(self, java_content):
        """
        从Java文件内容中提取代码片段
        :param java_content: Java文件内容
        :return: 代码片段列表
        """
        # 这里可以根据具体需求实现代码片段的提取逻辑
        # 例如：按方法分割、按类分割等
        # 简单示例：按空行分割
        segments = [seg.strip() for seg in java_content.split('\n\n') if seg.strip()]
        return segments

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

    def process_all_files(self):
        """
        处理目录下的所有Java文件
        """
        for root, _, files in os.walk(self.java_files_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    content = self.read_java_file(file_path)
                    if content:
                        segments = self.extract_code_segments(content)
                        for segment in segments:
                            # 这里需要实现或集成代码摘要生成的逻辑
                            non_bpe_summary = "非BPE摘要示例"  # 替换为实际的摘要生成逻辑
                            bpe_summary = "BPE摘要示例"  # 替换为实际的摘要生成逻辑
                            # 人工评分需要手动输入或其他方式获取
                            human_score = 0  # 替换为实际的评分获取逻辑
                            self.add_evaluation(segment, non_bpe_summary, bpe_summary, human_score)
