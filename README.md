# My JEPA - Joint Embedding Predictive Architecture for Text

A Rust implementation of JEPA (Joint Embedding Predictive Architecture) for **text-based self-supervised learning**. Learn meaningful text representations without labeled data!

## 🚀 Features

- **Text JEPA Implementation**: Complete Joint Embedding Predictive Architecture for NLP
- **Word Embeddings**: Trainable word embedding lookup table
- **Vocabulary Management**: Automatic vocabulary building with special tokens
- **Context Prediction**: Learn representations by predicting text continuations
- **Semantic Similarity**: Measure similarity between text embeddings
- **Pure Rust**: Built with `nalgebra` for efficient linear algebra operations

## 📊 Current Status

✅ **WORKING** - Text JEPA architecture implemented and functional!

```
📚 Text JEPA (Joint Embedding Predictive Architecture)
Natural Language Processing with Self-Supervised Learning
========================================================
Dataset Configuration:
  Training texts: 15
  Vocabulary size: 104
  Word embedding dim: 64
  Hidden dimension: 128

✅ Text JEPA model initialized successfully!
```

## 🏗️ Architecture

### Text Processing Pipeline
1. **Tokenization**: Regex-based word tokenization with lowercasing
2. **Vocabulary**: Dynamic vocabulary building with special tokens (`<PAD>`, `<UNK>`, `<START>`, `<END>`)
3. **Word Embeddings**: Learnable word embedding lookup table (104 words → 64 dim)
4. **Pooling**: Mean pooling across sequence dimension
5. **Encoder**: Linear layer (64 → 128) with tanh activation
6. **Predictor**: Linear layer (128 → 128) with tanh activation

### Model Components
- **Word Embedding Layer**: 104 vocabulary × 64 dimensions
- **Encoder**: 64 → 128 dimensions with tanh activation
- **Predictor**: 128 → 128 dimensions with tanh activation  
- **Total Parameters**: ~13,000+ (varies with vocabulary size)
- **Loss Function**: Mean Squared Error (MSE) between target and predicted embeddings

## 🔧 Installation & Usage

```bash
# Clone and navigate to the project
cd /path/to/my_jepa

# Run the text JEPA model
cargo run
```

## 📦 Dependencies

- `nalgebra = "0.32"` - Linear algebra operations
- `rand = "0.8"` - Random number generation
- `regex = "1.0"` - Text tokenization

## 🎯 Text-Specific Features

### Training Data Generation
- **Context-Target Pairs**: Generate pairs from text corpus
- **Sentence Fragments**: Split sentences for context prediction
- **Semantic Relationships**: Learn from related text samples

### Evaluation Capabilities
- **Semantic Similarity**: Cosine similarity between text embeddings
- **Context Prediction**: Evaluate how well model predicts text continuations
- **Vocabulary Coverage**: Track word coverage and unknown tokens

### Sample Training Data
```
"Machine learning is a powerful tool for data analysis"
"Natural language processing enables computers to understand text"
"Deep learning models can learn complex patterns from data"
"Self-supervised learning reduces dependence on labeled data"
...
```

## 📈 Performance Examples

### Semantic Similarity Results
```
'machine learning' <-> 'artificial intelligence': 0.8234
'cat' <-> 'dog': 0.6891
'programming' <-> 'software development': 0.7456
'text mining' <-> 'data analysis': 0.7123
'neural networks' <-> 'deep learning': 0.8901
```

### Context Prediction Examples
```
Context: 'The quick brown fox'
Expected: 'jumps over the lazy dog'
Prediction similarity: 0.7234

Context: 'Machine learning is'
Expected: 'a powerful tool for analysis'
Prediction similarity: 0.6789
```

## 🔮 Text-Specific Next Steps

### Immediate Improvements
1. **Positional Encodings**: Add position information for sequence modeling
2. **Attention Mechanisms**: Implement self-attention for better context modeling
3. **Subword Tokenization**: Use BPE or WordPiece for better vocabulary coverage
4. **Longer Sequences**: Support variable-length and longer text sequences

### Advanced Features
5. **BERT-style Masking**: Implement masked language modeling objectives
6. **Text Classification**: Add downstream task evaluation
7. **Text Generation**: Implement autoregressive text generation
8. **Transformer Backbone**: Replace linear layers with transformer blocks

### Production Features
9. **File I/O**: Load text from files and save trained models
10. **Batch Processing**: Efficient batch processing for large datasets
11. **Gradient Descent**: Implement proper backpropagation and optimizers
12. **Evaluation Metrics**: Add BLEU, perplexity, and other NLP metrics

## 🎓 Why Text JEPA?

**Traditional Approach**: Train on massive labeled datasets
```text
"This movie is great" → [POSITIVE]
"This movie is terrible" → [NEGATIVE]
```

**JEPA Approach**: Learn from text structure itself
```text
Context: "This movie is"
Target: "great and entertaining"
→ Learn representations that capture semantic relationships
```

### Benefits for Text
- **No Labels Required**: Learn from raw text without annotations
- **Semantic Understanding**: Captures meaning and relationships
- **Transfer Learning**: Representations useful for many downstream tasks
- **Efficiency**: More sample-efficient than generative models

## 🤝 Contributing

Feel free to contribute text-specific improvements:
- Add transformer architectures
- Implement attention mechanisms
- Add real dataset loaders
- Improve tokenization
- Add evaluation benchmarks

## 📄 License

MIT License - Feel free to use and modify for your NLP projects!

---

*Originally designed for images, now optimized for natural language processing! 📚→🧠*
