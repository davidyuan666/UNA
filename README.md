# BPE-UNA: Unified Neural Architecture with Enhanced BPE and VOLT Integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/bpe-una/badge/?version=latest)](https://bpe-una.readthedocs.io/en/latest/?badge=latest)

BPE-UNA implements a Unified Neural Architecture that combines Byte Pair Encoding (BPE) and Vocabulary Learning through Optimization (VOLT) for efficient code and natural language processing, with special support for Java and Python.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance](#performance)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

BPE-UNA provides an integrated approach for vocabulary construction that:
- Combines code and natural language tokens in a unified vocabulary
- Supports language-specific tokenization for Java and Python
- Optimizes vocabulary size using VOLT optimization
- Provides cross-modal semantic preservation
- Implements efficient evaluation metrics

## Features

- **Multiple Encoder Support**
  - Unified encoder for general purpose
  - Java-specific BPE encoder
  - Python-specific BPE encoder

- **Integrated BPE-VOLT Optimization**
  - Temperature-based token selection
  - Dynamic utility scoring
  - Cross-modal optimization

- **Language-Specific Features**
  - Java syntax awareness
  - Python indentation handling
  - Language-specific tokenization

- **Evaluation Metrics**
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)
  - Recall@K (K=1,5,10)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/davidyuan666/bpe-UNA.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Unified Encoder
```python
from unified_encoder import UnifiedEncoder

# Initialize encoder
encoder = UnifiedEncoder(
    vocab_size=8192,
    pct_bpe=0.2,
    volt_temp=1.0,
    modality_weights={'code': 0.6, 'nl': 0.4}
)

# Prepare data
code_data = ["def example(): pass", "class Test: ..."]
nl_data = ["This is a test", "Example documentation"]

# Fit and transform
encoder.fit(code_data, nl_data)
encoded = encoder.transform("def example():", modality='code')
```

### Using Language-Specific Encoders
```python
from java_bpe_encoder import JavaBPETokenizer
from python_bpe_encoder import PythonBPETokenizer

# Java encoder
java_encoder = JavaBPETokenizer(vocab_size=8192)
java_encoder.fit(java_code_data, nl_data)

# Python encoder
python_encoder = PythonBPETokenizer(vocab_size=8192)
python_encoder.fit(python_code_data, nl_data)
```

## Configuration

### Key Parameters

| Parameter | Description | Default | Applicable To |
|-----------|-------------|---------|---------------|
| `vocab_size` | Maximum vocabulary size | 8192 | All encoders |
| `pct_bpe` | Percentage for BPE tokens | 0.2 | All encoders |
| `volt_temp` | VOLT temperature | 1.0 | All encoders |
| `modality_weights` | Modality weights | {'code': 0.6, 'nl': 0.4} | UnifiedEncoder |

### Advanced Usage
```python
# Run comprehensive evaluation
from main import run_test

results = run_test()
# Results include MRR, NDCG, and Recall@K metrics
```

## Usage Examples

### Code Processing
```python
# Java code processing
java_code = """
public class Example {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
encoded_java = java_encoder.transform(java_code, modality='code')

# Python code processing
python_code = """
def add(a, b):
    return a + b
"""
encoded_python = python_encoder.transform(python_code, modality='code')
```

### Evaluation
```python
from main import evaluate_encoder

metrics = evaluate_encoder(
    encoder=encoder,
    code_corpus=code_data,
    nl_corpus=nl_data
)
print(f"MRR: {metrics['mrr']:.3f}")
print(f"NDCG: {metrics['ndcg']:.3f}")
```


## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes and commit (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bpe-una2023,
title={UNA: Unified Neural Architecture for Code-Related Tasks},
author={David Yuan},
journal={XX},
year={2024}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Issues**: Please use the [GitHub issue tracker](https://github.com/davidyuan666/bpe-UNA/issues)
- **Email**: wu.xiguanghua2014@gmail.com

---

## Acknowledgments

- Thanks to all contributors and researchers in the field
- Special thanks to the open-source community

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.

## Roadmap

- [ ] Support for additional programming languages
- [ ] Enhanced integration techniques
- [ ] Improved evaluation metrics
- [ ] GUI interface for visualization
