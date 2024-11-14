import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import json
import javalang

class JavaBPETokenizer:
    def __init__(self, vocab_size: int = 50000):
        """
        初始化Java专用的BPE分词器
        :param vocab_size: 目标词汇表大小
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        
        # 基础特殊标记
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<INDENT>': 4,
            '<DEDENT>': 5,
            '<NEWLINE>': 6,
        }
        
        # Java关键字标记
        self.java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
            'char', 'class', 'const', 'continue', 'default', 'do', 'double',
            'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
            'if', 'implements', 'import', 'instanceof', 'int', 'interface',
            'long', 'native', 'new', 'package', 'private', 'protected',
            'public', 'return', 'short', 'static', 'strictfp', 'super',
            'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
            'try', 'void', 'volatile', 'while'
        }
        
        self.keyword_tokens = {
            f'<{kw.upper()}>': i + len(self.special_tokens)
            for i, kw in enumerate(self.java_keywords)
        }
        
        # Java常用API标记
        self.common_api_tokens = {
            '<STRING>': 'String',
            '<SYSTEM>': 'System',
            '<LIST>': 'List',
            '<MAP>': 'Map',
            '<SET>': 'Set',
            '<ARRAY_LIST>': 'ArrayList',
            '<HASH_MAP>': 'HashMap',
            '<HASH_SET>': 'HashSet',
            '<EXCEPTION>': 'Exception',
            '<OBJECT>': 'Object',
        }
        
        # Java操作符标记
        self.operator_tokens = {
            '<PLUS>': '+',
            '<MINUS>': '-',
            '<MULT>': '*',
            '<DIV>': '/',
            '<MOD>': '%',
            '<EQ>': '==',
            '<NEQ>': '!=',
            '<LT>': '<',
            '<GT>': '>',
            '<LTE>': '<=',
            '<GTE>': '>=',
            '<ASSIGN>': '=',
            '<AND>': '&&',
            '<OR>': '||',
            '<NOT>': '!',
            '<BIT_AND>': '&',
            '<BIT_OR>': '|',
            '<BIT_XOR>': '^',
            '<BIT_NOT>': '~',
            '<DOT>': '.',
            '<COMMA>': ',',
            '<COLON>': ':',
            '<SEMICOLON>': ';',
            '<LPAREN>': '(',
            '<RPAREN>': ')',
            '<LBRACK>': '[',
            '<RBRACK>': ']',
            '<LBRACE>': '{',
            '<RBRACE>': '}',
        }

    def tokenize_java(self, code: str) -> List[str]:
        """
        使用javalang进行初步分词
        :param code: Java代码字符串
        :return: token列表
        """
        try:
            tokens = list(javalang.tokenizer.tokenize(code))
            return [token.value for token in tokens]
        except:
            # 处理不完整或有语法错误的代码片段
            return self._fallback_tokenize(code)

    def _fallback_tokenize(self, code: str) -> List[str]:
        """
        备用分词方法，用于处理不完整或有语法错误的代码
        :param code: 代码字符串
        :return: token列表
        """
        # 分割标识符和操作符
        pattern = r'([a-zA-Z_]\w*|[+\-*/=<>!&|^~.,;:(){}\[\]]|"[^"]*"|\'[^\']*\'|\d+)'
        return [token for token in re.findall(pattern, code) if token.strip()]

    def preprocess_code(self, code: str) -> str:
        """
        预处理Java代码
        :param code: 原始代码字符串
        :return: 预处理后的标记序列
        """
        tokens = self.tokenize_java(code)
        processed_tokens = []
        
        for token in tokens:
            # 处理关键字
            if token in self.java_keywords:
                processed_tokens.append(f'<{token.upper()}>')
            # 处理常用API
            elif token in self.common_api_tokens.values():
                api_token = next(k for k, v in self.common_api_tokens.items() if v == token)
                processed_tokens.append(api_token)
            # 处理操作符
            elif token in self.operator_tokens.values():
                op_token = next(k for k, v in self.operator_tokens.items() if v == token)
                processed_tokens.append(op_token)
            # 处理字符串字面量
            elif token.startswith('"') or token.startswith("'"):
                processed_tokens.append('<STRING_LIT>')
            # 处理数字字面量
            elif token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
                processed_tokens.append('<NUMBER_LIT>')
            else:
                processed_tokens.append(token)
                
        return ' '.join(processed_tokens)

    def get_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        从预处理后的文本中获取初始词汇表
        :param texts: 预处理后的文本列表
        :return: 词频字典
        """
        vocab = defaultdict(int)
        for text in texts:
            for token in text.split():
                vocab[token] += 1
        return vocab

    def get_pairs(self, word: str) -> Set[Tuple[str, str]]:
        """
        获取相邻token对
        :param word: 输入序列
        :return: token对集合
        """
        tokens = word.split()
        return set(zip(tokens[:-1], tokens[1:]))

    def train(self, code_samples: List[str]):
        """
        训练BPE模型
        :param code_samples: Java代码样本列表
        """
        # 预处理所有代码样本
        processed_texts = [self.preprocess_code(code) for code in code_samples]
        
        # 获取初始词汇表
        vocab = self.get_vocab(processed_texts)
        
        # 计算需要合并的次数
        num_merges = self.vocab_size - (len(self.special_tokens) + 
                                      len(self.keyword_tokens) + 
                                      len(self.common_api_tokens) + 
                                      len(self.operator_tokens))
        
        for i in range(num_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for pair in self.get_pairs(word):
                    pairs[pair] += freq
                    
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token
            
            # 更新词汇表
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = word.replace(' '.join(best_pair), new_token)
                new_vocab[new_word] = freq
            vocab = new_vocab

        # 构建最终词汇表
        self.vocab = {**self.special_tokens, **self.keyword_tokens,
                     **self.common_api_tokens, **{k: i + len(self.special_tokens) + 
                     len(self.keyword_tokens) + len(self.common_api_tokens)
                     for i, k in enumerate(vocab.keys())}}

    def encode(self, code: str) -> List[int]:
        """
        将Java代码编码为token ID序列
        :param code: Java代码
        :return: token ID列表
        """
        processed_code = self.preprocess_code(code)
        tokens = processed_code.split()
        encoded = []
        
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                # 应用合并规则
                current_token = ' '.join(list(token))
                while True:
                    pairs = self.get_pairs(current_token)
                    if not pairs:
                        break
                    
                    mergeable_pair = None
                    for pair in pairs:
                        if pair in self.merges:
                            mergeable_pair = pair
                            break
                            
                    if not mergeable_pair:
                        break
                        
                    current_token = current_token.replace(
                        ' '.join(mergeable_pair),
                        self.merges[mergeable_pair]
                    )
                
                if current_token in self.vocab:
                    encoded.append(self.vocab[current_token])
                else:
                    encoded.append(self.vocab['<UNK>'])
                    
        return encoded

    def decode(self, tokens: List[int]) -> str:
        """
        将token ID序列解码为Java代码
        :param tokens: token ID列表
        :return: 解码后的代码
        """
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded = []
        
        for token in tokens:
            if token in inv_vocab:
                token_str = inv_vocab[token]
                if token_str.startswith('<') and token_str.endswith('>'):
                    # 处理特殊标记
                    if token_str in self.operator_tokens:
                        decoded.append(self.operator_tokens[token_str])
                    elif token_str in self.common_api_tokens:
                        decoded.append(self.common_api_tokens[token_str])
                    else:
                        # 移除尖括号并还原关键字
                        pure_token = token_str[1:-1].lower()
                        decoded.append(pure_token)
                else:
                    decoded.append(token_str)
                    
        return ' '.join(decoded)

    def save(self, path: str):
        """
        保存模型到文件
        :param path: 保存路径
        """
        model_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'keyword_tokens': self.keyword_tokens,
            'common_api_tokens': self.common_api_tokens,
            'operator_tokens': self.operator_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=