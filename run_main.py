# -*- coding = utf-8 -*-
# @time:2024/10/27 21:26
# Author:david yuan
# @File:main.py
# @Software:MacauUST&VeSync

from unified_encoder import UnifiedEncoder
from java_bpe_encoder import JavaBPETokenizer
from python_bpe_encoder import PythonBPETokenizer
from human_evaluation import JavaCodeEvaluator
from python_evaluation import PythonCodeEvaluator
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    """Run test cases for different encoders with performance metrics"""
    # Load datasets
    datasets = {
        'codesearchnet': {
            'code_vocab_size': 77345,
            'nl_vocab_size': 28856,
            'uni_vocab_size': 2054,
            'code_corpus': {
                'python': load_codesearchnet_code('python'),
                'java': load_codesearchnet_code('java')
            },
            'nl_corpus': load_codesearchnet_nl()
        },
        'funcom': {
            'code_vocab_size': 241099,
            'nl_vocab_size': 96861,
            'uni_vocab_size': 2076,
            'code_corpus': {
                'java': load_funcom_code()
            },
            'nl_corpus': load_funcom_nl()
        }
    }

    encoders = {
        'unified': UnifiedEncoder,
        'java': JavaBPETokenizer,
        'python': PythonBPETokenizer
    }

    results = {}
    
    # Test on different datasets
    for dataset_name, dataset in datasets.items():
        logger.info(f"\nTesting on {dataset_name} dataset:")
        results[dataset_name] = {}
        
        vocab_modes = ['TV', 'OV', 'Uni-Vocab']
        for mode in vocab_modes:
            logger.info(f"\nTesting {mode} vocabulary mode:")
            results[dataset_name][mode] = {}
            
            # Set vocabulary size based on mode
            if mode == 'TV':
                vocab_size = dataset['code_vocab_size'] + dataset['nl_vocab_size']
            elif mode == 'OV':
                vocab_size = max(dataset['code_vocab_size'], dataset['nl_vocab_size'])
            else:  # Uni-Vocab
                vocab_size = dataset['uni_vocab_size']
            
            for encoder_name, encoder_class in encoders.items():
                logger.info(f"\nTesting {encoder_name} encoder:")
                
                if encoder_name == 'java' and 'java' not in dataset['code_corpus']:
                    continue
                if encoder_name == 'python' and 'python' not in dataset['code_corpus']:
                    continue
                
                encoder = encoder_class(
                    vocab_size=vocab_size,
                    pct_bpe=0.2,
                    volt_temp=1.0,
                    modality_weights={'code': 0.6, 'nl': 0.4},
                    vocab_mode=mode
                )
                
                if encoder_name == 'java':
                    code_corpus = dataset['code_corpus']['java']
                elif encoder_name == 'python':
                    code_corpus = dataset['code_corpus']['python']
                else:
                    code_corpus = []
                    for lang_corpus in dataset['code_corpus'].values():
                        code_corpus.extend(lang_corpus)
                
                # Fit the encoder
                logger.info("Fitting encoder...")
                encoder.fit(code_corpus, dataset['nl_corpus'])
                
                # Evaluate performance
                metrics = evaluate_encoder(
                    encoder=encoder,
                    code_corpus=code_corpus,
                    nl_corpus=dataset['nl_corpus']
                )
                
                results[dataset_name][mode][encoder_name] = metrics
                
                logger.info(f"Vocabulary Size: {vocab_size}")
                logger.info(f"MRR Score: {metrics['mrr']:.3f}")
                logger.info(f"NDCG Score: {metrics['ndcg']:.3f}")
                logger.info(f"R@1: {metrics['r@1']:.3f}")
                logger.info(f"R@5: {metrics['r@5']:.3f}")
                logger.info(f"R@10: {metrics['r@10']:.3f}")

    return results

def evaluate_encoder(encoder, code_corpus: List[str], nl_corpus: List[str]) -> Dict[str, float]:
    """Evaluate encoder performance with multiple metrics"""
    # Encode both corpora
    encoded_code = [encoder.transform(code, modality='code') for code in code_corpus]
    encoded_nl = [encoder.transform(nl, modality='nl') for nl in nl_corpus]
    
    # Convert to tensors
    code_vectors = torch.tensor(encoded_code)
    nl_vectors = torch.tensor(encoded_nl)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(code_vectors, nl_vectors)
    
    # Calculate metrics
    metrics = {
        'mrr': calculate_mrr(similarity_matrix),
        'ndcg': calculate_ndcg(similarity_matrix),
        'r@1': calculate_recall_at_k(similarity_matrix, k=1),
        'r@5': calculate_recall_at_k(similarity_matrix, k=5),
        'r@10': calculate_recall_at_k(similarity_matrix, k=10)
    }
    
    return metrics

def calculate_mrr(similarity_matrix: np.ndarray) -> float:
    """Calculate Mean Reciprocal Rank"""
    rankings = (-similarity_matrix).argsort(axis=1)
    correct_rankings = np.where(rankings == np.arange(len(rankings))[:, None])[1]
    mrr = np.mean(1 / (correct_rankings + 1))
    return float(mrr)

def calculate_ndcg(similarity_matrix: np.ndarray) -> float:
    """Calculate Normalized Discounted Cumulative Gain"""
    rankings = (-similarity_matrix).argsort(axis=1)
    ideal_rankings = np.arange(similarity_matrix.shape[1])
    
    dcg = np.sum(1 / np.log2(rankings + 2), axis=1)
    idcg = np.sum(1 / np.log2(ideal_rankings + 2))
    
    ndcg = np.mean(dcg / idcg)
    return float(ndcg)

def calculate_recall_at_k(similarity_matrix: np.ndarray, k: int) -> float:
    """Calculate Recall@K"""
    rankings = (-similarity_matrix).argsort(axis=1)
    correct_at_k = (rankings[:, :k] == np.arange(len(rankings))[:, None]).any(axis=1)
    recall = correct_at_k.mean()
    return float(recall)

def load_codesearchnet_code(language: str) -> List[str]:
    """Load code corpus from CodeSearchNet Challenge dataset
    
    Args:
        language: Programming language ('java' or 'python')
    Returns:
        List of code snippets
    """
    import json
    import os
    
    code_samples = []
    
    # 构建文件路径
    if language == 'java':
        file_path = os.path.join(os.getcwd(), 'dataset', 'java_sample.jsonl')
    elif language == 'python':
        file_path = os.path.join(os.getcwd(), 'dataset', 'python_sample.jsonl')
    else:
        logger.error(f"Unsupported language: {language}")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'code' in data:
                        code_samples.append(data['code'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line: {line[:100]}...")
                    continue
                    
        logger.info(f"Loaded {len(code_samples)} {language} code samples")
        return code_samples
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading {language} code samples: {str(e)}")
        return []

def load_codesearchnet_nl() -> List[str]:
    """Load natural language corpus from CodeSearchNet Challenge dataset"""
    import json
    import os
    
    nl_samples = []
    file_path = os.path.join(os.getcwd(), 'dataset', 'java_sample.jsonl')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'docstring' in data:
                        nl_samples.append(data['docstring'])
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"Loaded {len(nl_samples)} natural language descriptions")
        return nl_samples
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading natural language samples: {str(e)}")
        return []

def load_funcom_code() -> List[str]:
    """Load code corpus from funcom dataset
    
    Returns:
        List of Java code snippets from the funcom dataset
    """
    import json
    import os
    
    code_samples = []
    file_path = os.path.join(os.getcwd(), 'dataset', 'funcom_sample.jsonl')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # funcom dataset typically has 'function' field for code
                    if 'function' in data:
                        code_samples.append(data['function'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse funcom code line: {line[:100]}...")
                    continue
                    
        logger.info(f"Loaded {len(code_samples)} funcom code samples")
        return code_samples
        
    except FileNotFoundError:
        logger.error(f"Funcom file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading funcom code samples: {str(e)}")
        return []

def load_funcom_nl() -> List[str]:
    """Load natural language corpus from funcom dataset
    
    Returns:
        List of natural language comments from the funcom dataset
    """
    import json
    import os
    
    nl_samples = []
    file_path = os.path.join(os.getcwd(), 'dataset', 'funcom_sample.jsonl')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # funcom dataset typically has 'comment' field for natural language
                    if 'comment' in data:
                        nl_samples.append(data['comment'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse funcom comment line: {line[:100]}...")
                    continue
                    
        logger.info(f"Loaded {len(nl_samples)} funcom natural language samples")
        return nl_samples
        
    except FileNotFoundError:
        logger.error(f"Funcom file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading funcom natural language samples: {str(e)}")
        return []

def save_results(results: Dict, filepath: str):
    """Save evaluation results to file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def run_human_evaluation():
    """Run human evaluation for Java and Python code using LLM-based summarization
    
    This function:
    1. Loads code samples from JSONL files
    2. Samples 100 snippets for each language
    3. Generates summaries using both normal and BPE encoding
    4. Exports results to CSV files
    """
    import os
    import random
    from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
    
    # Initialize the code summarization model
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
    
    # Java evaluation
    logger.info("Starting Java code evaluation...")
    java_files_path = os.path.join(os.getcwd(), 'dataset', 'java_sample.jsonl')
    java_evaluator = JavaCodeEvaluator(
        java_files_path=java_files_path,
        model=model,
        tokenizer=tokenizer
    )
    
    # Load and sample Java code
    java_codes = load_codesearchnet_code('java')
    if len(java_codes) > 100:
        sampled_java_codes = random.sample(java_codes, 100)
        logger.info(f"Sampled 100 Java code snippets from {len(java_codes)} total snippets")
    else:
        sampled_java_codes = java_codes
        logger.warning(f"Found only {len(java_codes)} Java snippets, using all available")
    
    # Process Java samples
    for idx, code in enumerate(sampled_java_codes, 1):
        logger.info(f"Processing Java sample {idx}/100...")
        
        # Generate normal summary
        normal_summary = java_evaluator.generate_summary(code)
        
        # Generate BPE-based summary
        bpe_encoded = java_evaluator.encode_with_bpe(code)
        bpe_summary = java_evaluator.generate_summary(bpe_encoded)
        
        # Add to evaluation results
        java_evaluator.add_evaluation(
            code_segment=code,
            non_bpe_summary=normal_summary,
            bpe_summary=bpe_summary,
            human_score=None  # To be filled by human evaluators
        )
    
    # Export Java results
    java_output_path = os.path.join(os.getcwd(), 'results', 'java_evaluation_results.csv')
    os.makedirs(os.path.dirname(java_output_path), exist_ok=True)
    java_evaluator.export_to_csv(java_output_path)
    logger.info(f"Java evaluation results saved to {java_output_path}")
    
    # Python evaluation
    logger.info("\nStarting Python code evaluation...")
    python_files_path = os.path.join(os.getcwd(), 'dataset', 'python_sample.jsonl')
    python_evaluator = PythonCodeEvaluator(
        python_files_path=python_files_path,
        model=model,
        tokenizer=tokenizer
    )
    
    # Load and sample Python code
    python_codes = load_codesearchnet_code('python')
    if len(python_codes) > 100:
        sampled_python_codes = random.sample(python_codes, 100)
        logger.info(f"Sampled 100 Python code snippets from {len(python_codes)} total snippets")
    else:
        sampled_python_codes = python_codes
        logger.warning(f"Found only {len(python_codes)} Python snippets, using all available")
    
    # Process Python samples
    for idx, code in enumerate(sampled_python_codes, 1):
        logger.info(f"Processing Python sample {idx}/100...")
        
        # Generate normal summary
        normal_summary = python_evaluator.generate_summary(code)
        
        # Generate BPE-based summary
        bpe_encoded = python_evaluator.encode_with_bpe(code)
        bpe_summary = python_evaluator.generate_summary(bpe_encoded)
        
        # Add to evaluation results
        python_evaluator.add_evaluation(
            code_segment=code,
            non_bpe_summary=normal_summary,
            bpe_summary=bpe_summary,
            human_score=None  # To be filled by human evaluators
        )
    
    # Export Python results
    python_output_path = os.path.join(os.getcwd(), 'results', 'python_evaluation_results.csv')
    os.makedirs(os.path.dirname(python_output_path), exist_ok=True)
    python_evaluator.export_to_csv(python_output_path)
    logger.info(f"Python evaluation results saved to {python_output_path}")

if __name__ == "__main__":
    results = run_test()
    save_results(results, 'encoder_evaluation_results.json')
    run_human_evaluation()