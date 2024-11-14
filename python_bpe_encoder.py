import tokenize
from io import StringIO
import keyword
import builtins
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Set, Optional
import json

class PythonBPETokenizer:
    def __init__(self, vocab_size: int = 50000):
        """
        初始化Python专用的BPE分词器
        :param vocab_size: 目标词汇表大小
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        
        # Python语言特定的标记
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<INDENT>': 4,
            '<DEDENT>': 5,
            '<NEWLINE>': 6,
        }
        
        # Python关键字标记
        self.keyword_tokens = {
            f'<{kw.upper()}>': i + len(self.special_tokens)
            for i, kw in enumerate(keyword.kwlist)
        }
        
        # Python内置函数标记
        self.builtin_tokens = {
            f'<BUILTIN_{func.upper()}>': i + len(self.special_tokens) + len(self.keyword_tokens)
            for i, func in enumerate(dir(builtins))
            if not func.startswith('_')
        }
        
        # Python操作符标记
        self.operator_tokens = {
            '<PLUS>': '+',
            '<MINUS>': '-',
            '<MULT>': '*',
            '<DIV>': '/',
            '<FLOORDIV>': '//',
            '<MOD>': '%',
            '<POW>': '**',
            '<LSHIFT>': '<<',
            '<RSHIFT>': '>>',
            '<AND>': '&',
            '<OR>': '|',
            '<XOR>': '^',
            '<EQ>': '==',
            '<NEQ>': '!=',
            '<LT>': '<',
            '<GT>': '>',
            '<LTE>': '<=',
            '<GTE>': '>=',
            '<ASSIGN>': '=',
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

    def tokenize_python(self, code: str) -> List[Tuple[int, str]]:
        """
        使用Python的tokenize模块进行初步分词
        :param code: Python代码字符串
        :return: token列表
        """
        tokens = []
        try:
            for tok in tokenize.generate_tokens(StringIO(code).readline):
                tokens.append((tok.type, tok.string))
        except tokenize.TokenError:
            # 处理不完整的代码片段
            pass
        return tokens

    def preprocess_code(self, code: str) -> str:
        """
        预处理Python代码
        :param code: 原始代码字符串
        :return: 预处理后的标记序列
        """
        tokens = self.tokenize_python(code)
        processed_tokens = []
        indent_level = 0
        
        for tok_type, tok_string in tokens:
            if tok_type == tokenize.INDENT:
                indent_level += 1
                processed_tokens.append('<INDENT>')
            elif tok_type == tokenize.DEDENT:
                indent_level -= 1
                processed_tokens.append('<DEDENT>')
            elif tok_type == tokenize.NEWLINE:
                processed_tokens.append('<NEWLINE>')
            elif tok_type == tokenize.NAME:
                if tok_string in keyword.kwlist:
                    processed_tokens.append(f'<{tok_string.upper()}>')
                elif tok_string in dir(builtins) and not tok_string.startswith('_'):
                    processed_tokens.append(f'<BUILTIN_{tok_string.upper()}>')
                else:
                    processed_tokens.append(tok_string)
            elif tok_type == tokenize.OP:
                op_token = next((k for k, v in self.operator_tokens.items() 
                               if v == tok_string), tok_string)
                processed_tokens.append(op_token)
            elif tok_type == tokenize.STRING:
                # 处理字符串字面量
                processed_tokens.append('<STRING_LIT>')
            elif tok_type == tokenize.NUMBER:
                # 处理数字字面量
                processed_tokens.append('<NUMBER_LIT>')
            elif tok_type == tokenize.COMMENT:
                # 处理注释
                processed_tokens.append('<COMMENT>')
            elif tok_string.strip():
                processed_tokens.append(tok_string)
                
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
        :param code_samples: Python代码样本列表
        """
        # 预处理所有代码样本
        processed_texts = [self.preprocess_code(code) for code in code_samples]
        
        # 获取初始词汇表
        vocab = self.get_vocab(processed_texts)
        
        # 计算需要合并的次数
        num_merges = self.vocab_size - (len(self.special_tokens) + 
                                      len(self.keyword_tokens) + 
                                      len(self.builtin_tokens) + 
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
                     **self.builtin_tokens, **{k: i + len(self.special_tokens) + 
                     len(self.keyword_tokens) + len(self.builtin_tokens)
                     for i, k in enumerate(vocab.keys())}}

    def encode(self, code: str) -> List[int]:
        """
        将Python代码编码为token ID序列
        :param code: Python代码
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
        将token ID序列解码为Python代码
        :param tokens: token ID列表
        :return: 解码后的代码
        """
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded = []
        indent_level = 0
        
        for token in tokens:
            if token in inv_vocab:
                token_str = inv_vocab[token]
                if token_str == '<INDENT>':
                    indent_level += 1
                    decoded.append('    ' * indent_level)
                elif token_str == '<DEDENT>':
                    indent_level -= 1
                elif token_str == '<NEWLINE>':
                    decoded.append('\n' + '    ' * indent_level)
                elif token_str.startswith('<') and token_str.endswith('>'):
                    # 处理特殊标记
                    if token_str in self.operator_tokens:
                        decoded.append(self.operator_tokens[token_str])
                    else:
                        # 移除尖括号并还原关键字/内置函数
                        pure_token = token_str[1:-1].lower()
                        if pure_token.startswith('builtin_'):
                            pure_token = pure_token[8:].lower()
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
            'builtin_tokens': self.builtin_tokens,
            'operator_tokens': self.operator_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """
        从文件加载模型
        :param path: 模型文件路径
        """
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        self.vocab = model_data['vocab']
        self.merges = model_data['merges']
        self.special_tokens = model_data['special_tokens']
        self.keyword_tokens = model_data['keyword_tokens']
        self.builtin_tokens = model_data['builtin_tokens']
        self.operator_tokens = model_data['operator_tokens']