# -*- coding = utf-8 -*-
# @time:2024/10/26 06:30
# Author:david yuan
# @File:unified_encoder.py
# @Software:MacauUST&VeSync

from collections import Counter
import torch
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import toolz
import json

try:
    from typing import Dict, Iterable, Callable, List, Any, Iterator
except ImportError:
    pass

DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'

class UnifiedEncoder:
    def __init__(self, vocab_size=8192, pct_bpe=0.2, 
                 word_tokenizer=None, silent=True,
                 ngram_min=2, ngram_max=2, 
                 volt_temp=1.0, # Temperature parameter for VOLT
                 modality_weights={'code': 0.5, 'nl': 0.5}, # Weights for different modalities
                 strict=False, lowercase=True,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, 
                 UNK=DEFAULT_UNK, PAD=DEFAULT_PAD,
                 **kwargs):
        """
        Unified Neural Architecture encoder combining BPE and VOLT
        
        Args:
            vocab_size: Maximum vocabulary size
            pct_bpe: Percentage of vocab to reserve for BPE tokens
            volt_temp: Temperature parameter for VOLT optimization
            modality_weights: Weights for different modalities in unified vocab
        """
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')
            
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.volt_temp = volt_temp
        self.modality_weights = modality_weights
        
        # Special tokens
        self.EOW = EOW
        self.SOW = SOW
        self.UNK = UNK
        self.PAD = PAD
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        
        # Initialize basic components
        self.word_tokenizer = word_tokenizer or wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}
        self.bpe_vocab = {}
        self._progress_bar = iter if silent else tqdm
        
        # BPE parameters
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict
        self.lowercase = lowercase
        
        # Calculate vocab sizes
        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len([self.UNK, self.PAD])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        
        # Initialize vocabularies
        self.unified_vocab = {}
        self.modality_specific_vocabs = {
            'code': Counter(),
            'nl': Counter()
        }
        self.inverse_word_vocab = {}
        self.inverse_bpe_vocab = {}

    def _count_tokens(self, corpus, modality):
        """Count tokens in corpus for specific modality"""
        counts = Counter()
        for sample in self._progress_bar(corpus):
            tokens = self.tokenize(sample)
            counts.update(tokens)
            self.modality_specific_vocabs[modality].update(tokens)
        return counts

    def learn_unified_vocab(self, code_corpus, nl_corpus):
        """Learn unified vocabulary using VOLT optimization"""
        # Count tokens in both modalities
        code_counts = self._count_tokens(code_corpus, 'code')
        nl_counts = self._count_tokens(nl_corpus, 'nl')
        
        # Combine counts with modality weights
        combined_counts = Counter()
        for token, count in code_counts.items():
            combined_counts[token] += count * self.modality_weights['code']
        for token, count in nl_counts.items():
            combined_counts[token] += count * self.modality_weights['nl']
            
        # Apply VOLT optimization
        vocab = self._volt_optimize(combined_counts)
        return vocab

    def _volt_optimize(self, token_counts):
        """Implement VOLT optimization for vocabulary selection"""
        # Convert counts to probabilities
        total_count = sum(token_counts.values())
        probs = {token: count/total_count for token, count in token_counts.items()}
        
        # Apply temperature scaling
        scaled_probs = {
            token: np.exp(np.log(prob) / self.volt_temp) 
            for token, prob in probs.items()
        }
        
        # Select top tokens based on scaled probabilities
        sorted_tokens = sorted(scaled_probs.items(), key=lambda x: x[1], reverse=True)
        return {
            token: idx 
            for idx, (token, _) in enumerate(sorted_tokens[:self.vocab_size])
        }

    def byte_pair_counts(self, words):
        """Count byte pair frequencies"""
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]
                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count
            yield bp_counts

    def _learn_bpe(self, tokens):
        """Learn BPE vocabulary for OOV tokens"""
        # Initialize with character vocabulary
        vocab = Counter()
        for token in tokens:
            chars = ' '.join(list(token))
            vocab.update(chars.split())
            
        # Perform merges
        merges = {}
        while len(vocab) < self.bpe_vocab_size:
            pairs = self._get_merge_candidates(tokens)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)
            
            # Update vocabulary
            vocab[new_token] = pairs[best_pair]
            merges[best_pair] = new_token
            
            # Update tokens
            tokens = [self._merge_pair(token, best_pair, new_token) for token in tokens]
            
        return vocab

    def fit(self, code_data, nl_data):
        """Fit the encoder on both code and natural language data"""
        # Learn unified vocabulary
        self.unified_vocab = self.learn_unified_vocab(code_data, nl_data)
        
        # Learn BPE vocab for out-of-vocabulary tokens
        remaining_tokens = self._get_oov_tokens(code_data, nl_data)
        self.bpe_vocab = self._learn_bpe(remaining_tokens)
        
        # Create inverse mappings
        self.inverse_word_vocab = {v: k for k, v in self.word_vocab.items()}
        self.inverse_bpe_vocab = {v: k for k, v in self.bpe_vocab.items()}

    def transform(self, text, modality='nl', reverse=False, fixed_length=None):
        """Transform text to token indices with modality-specific handling"""
        direction = -1 if reverse else 1
        tokens = self.tokenize(text.lower().strip() if self.lowercase else text.strip())
        encoded = []
        in_subword = False
        
        for token in tokens:
            if in_subword:
                if token in self.bpe_vocab:
                    if token == self.EOW:
                        in_subword = False
                    encoded.append(self.bpe_vocab[token])
                else:
                    encoded.append(self.word_vocab[self.UNK])
            else:
                if token == self.SOW:
                    in_subword = True
                    encoded.append(self.bpe_vocab[token])
                else:
                    if token in self.unified_vocab:
                        encoded.append(self.unified_vocab[token])
                    elif token in self.bpe_vocab:
                        encoded.append(self.bpe_vocab[token])
                    else:
                        subwords = self.subword_tokenize(token)
                        encoded.extend(self.bpe_vocab.get(sw, self.bpe_vocab[self.UNK]) 
                                     for sw in subwords)

        if fixed_length is not None:
            encoded = encoded[:fixed_length]
            while len(encoded) < fixed_length:
                encoded.append(self.word_vocab[self.PAD])
                
        return encoded[::direction]

    def inverse_transform(self, rows):
        """Turns token indexes back into space-joined text"""
        for row in rows:
            words = []
            rebuilding_word = False
            current_word = ''
            
            for idx in row:
                if self.inverse_bpe_vocab.get(idx) == self.SOW:
                    if rebuilding_word and self.strict:
                        raise ValueError('Encountered second SOW token before EOW.')
                    rebuilding_word = True

                elif self.inverse_bpe_vocab.get(idx) == self.EOW:
                    if not rebuilding_word and self.strict:
                        raise ValueError('Encountered EOW without matching SOW.')
                    rebuilding_word = False
                    words.append(current_word)
                    current_word = ''

                elif rebuilding_word and (idx in self.inverse_bpe_vocab):
                    current_word += self.inverse_bpe_vocab[idx]

                elif rebuilding_word and (idx in self.inverse_word_vocab):
                    current_word += self.inverse_word_vocab[idx]

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx])

                elif idx in self.inverse_bpe_vocab:
                    if self.strict:
                        raise ValueError("Found BPE index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_bpe_vocab[idx])

                else:
                    raise ValueError("Got index {} that was not in word or BPE vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')

    def save(self, outpath, dont_warn=False, encoding=None, ensure_ascii=True, indent=2):
        """Serializes and saves encoder to provided path"""
        if self.custom_tokenizer and not dont_warn:
            print("WARNING! You've specified a non-default tokenizer. You'll need to reassign it when you load the model!")
            
        model_data = {
            'unified_vocab': self.unified_vocab,
            'bpe_vocab': self.bpe_vocab,
            'word_vocab': self.word_vocab,
            'modality_specific_vocabs': self.modality_specific_vocabs,
            'kwargs': {
                'vocab_size': self.vocab_size,
                'pct_bpe': self.pct_bpe,
                'volt_temp': self.volt_temp,
                'modality_weights': self.modality_weights,
                'silent': self._progress_bar is iter,
                'ngram_min': self.ngram_min,
                'ngram_max': self.ngram_max,
                'strict': self.strict,
                'lowercase': self.lowercase,
                'EOW': self.EOW,
                'SOW': self.SOW,
                'UNK': self.UNK,
                'PAD': self.PAD,
            }
        }
        
        with open(outpath, 'w', encoding=encoding) as outfile:
            json.dump(model_data, outfile, ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def load(cls, path):
        """Load encoder from saved file"""
        with open(path) as infile:
            model_data = json.load(infile)
            
        encoder = cls(**model_data['kwargs'])
        encoder.unified_vocab = model_data['unified_vocab']
        encoder.bpe_vocab = model_data['bpe_vocab']
        encoder.word_vocab = model_data['word_vocab']
        encoder.modality_specific_vocabs = model_data['modality_specific_vocabs']
        
        # Recreate inverse mappings
        encoder.inverse_word_vocab = {v: k for k, v in encoder.word_vocab.items()}
        encoder.inverse_bpe_vocab = {v: k for k, v in encoder.bpe_vocab.items()}
        
        return encoder