from collections import Counter
import torch
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import toolz
import json

class UnifiedEncoder:
    def __init__(self, vocab_size=8192, pct_bpe=0.2, 
                 word_tokenizer=None, silent=True,
                 ngram_min=2, ngram_max=2, 
                 volt_temp=1.0, # Temperature parameter for VOLT
                 modality_weights={'code': 0.5, 'nl': 0.5}, # Weights for different modalities
                 **kwargs):
        """
        Unified Neural Architecture encoder combining BPE and VOLT
        
        Args:
            vocab_size: Maximum vocabulary size
            pct_bpe: Percentage of vocab to reserve for BPE tokens
            volt_temp: Temperature parameter for VOLT optimization
            modality_weights: Weights for different modalities in unified vocab
        """
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.volt_temp = volt_temp
        self.modality_weights = modality_weights
        
        # Initialize basic components similar to original encoder
        self.word_tokenizer = word_tokenizer or wordpunct_tokenize
        self.word_vocab = {}
        self.bpe_vocab = {}
        self._progress_bar = iter if silent else tqdm
        
        # New components for unified vocabulary
        self.unified_vocab = {}
        self.modality_specific_vocabs = {
            'code': Counter(),
            'nl': Counter()
        }

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

    def fit(self, code_data, nl_data):
        """Fit the encoder on both code and natural language data"""
        # Learn unified vocabulary
        self.unified_vocab = self.learn_unified_vocab(code_data, nl_data)
        
        # Learn BPE vocab for out-of-vocabulary tokens
        remaining_tokens = self._get_oov_tokens(code_data, nl_data)
        self.bpe_vocab = self._learn_bpe(remaining_tokens)

    def _get_oov_tokens(self, code_data, nl_data):
        """Get tokens not in unified vocabulary"""
        all_tokens = set()
        for text in code_data + nl_data:
            tokens = self.tokenize(text)
            all_tokens.update(t for t in tokens if t not in self.unified_vocab)
        return list(all_tokens)

    def transform(self, text, modality='nl'):
        """Transform text to token indices with modality-specific handling"""
        tokens = self.tokenize(text)
        encoded = []
        
        for token in tokens:
            if token in self.unified_vocab:
                encoded.append(self.unified_vocab[token])
            elif token in self.bpe_vocab:
                encoded.append(self.bpe_vocab[token])
            else:
                # Handle OOV tokens
                subwords = self.subword_tokenize(token)
                encoded.extend(self.bpe_vocab.get(sw, self.bpe_vocab['<unk>']) 
                             for sw in subwords)
                
        return encoded


    def _integrate_bpe_volt(self, token_counts, num_merges):
        """Integrate BPE and VOLT optimization for merge operations
        
        Args:
            token_counts: Counter of initial tokens
            num_merges: Number of merge operations to perform
            
        Returns:
            Dictionary mapping merged tokens to their indices
        """
        # Initialize with character-level tokens
        vocab = {ch: idx for idx, ch in enumerate(set(''.join(token_counts.keys())))}
        merges = {}
        
        # Priority queue for merge candidates
        from heapq import heappush, heappop
        merge_queue = []
        
        # Track merge statistics
        merge_stats = {
            'vocab_size': [],
            'performance': []
        }
        
        for i in range(num_merges):
            # Get merge candidates from BPE
            pairs = self._get_merge_candidates(token_counts)
            
            # Score candidates using VOLT utility function
            for pair, freq in pairs.items():
                utility = self._calculate_utility(pair, freq, token_counts)
                heappush(merge_queue, (-utility, pair))
                
            # Execute highest utility merge
            if merge_queue:
                _, best_pair = heappop(merge_queue)
                self._execute_merge(best_pair, token_counts, vocab, merges)
                
                # Update statistics
                merge_stats['vocab_size'].append(len(vocab))
                
                # Check stopping criteria
                if self._should_stop(merge_stats):
                    break
                    
        return vocab, merges

    def _calculate_utility(self, pair, freq, token_counts):
        """Calculate utility score for a merge candidate using VOLT
        
        Args:
            pair: Tuple of tokens to be merged
            freq: Frequency of the pair
            token_counts: Current token counts
            
        Returns:
            Utility score combining frequency and semantic importance
        """
        # Frequency score
        f_score = freq / sum(token_counts.values())
        
        # Semantic importance score (e.g., based on cross-modal coverage)
        s_score = self._semantic_importance(pair)
        
        # Cross-modal coverage score
        c_score = self._cross_modal_coverage(pair)
        
        # Combine scores with weights from initialization
        utility = (self.modality_weights['code'] * f_score + 
                0.3 * s_score + 
                0.3 * c_score)
                
        return utility

    def _should_stop(self, merge_stats):
        """Determine if merge operations should stop based on statistics
        
        Args:
            merge_stats: Dictionary tracking merge statistics
            
        Returns:
            Boolean indicating whether to stop merging
        """
        if len(merge_stats['vocab_size']) < 2:
            return False
            
        # Check vocabulary growth rate
        vocab_growth = (merge_stats['vocab_size'][-1] - 
                    merge_stats['vocab_size'][-2]) / merge_stats['vocab_size'][-2]
                    
        # Stop if vocabulary growth is minimal
        if vocab_growth < 0.001:
            return True
            
        # Stop if vocabulary size exceeds limit
        if merge_stats['vocab_size'][-1] >= self.vocab_size:
            return True
            
        return False

    def _execute_merge(self, pair, token_counts, vocab, merges):
        """Execute a merge operation and update relevant data structures
        
        Args:
            pair: Tuple of tokens to merge
            token_counts: Current token counts
            vocab: Current vocabulary
            merges: Record of merge operations
        """
        new_token = ''.join(pair)
        
        # Add to vocabulary if not present
        if new_token not in vocab:
            vocab[new_token] = len(vocab)
            
        # Record merge operation
        merges[pair] = new_token
        
        # Update token counts
        token_counts[new_token] = token_counts[pair]
        del token_counts[pair]

