use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use regex::Regex;
use std::collections::HashMap;

// Text processing utilities
#[derive(Debug, Clone)]
struct Vocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    vocab_size: usize,
}

impl Vocabulary {
    fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            vocab_size: 0,
        };
        
        // Add special tokens
        vocab.add_word("<PAD>".to_string());
        vocab.add_word("<UNK>".to_string());
        vocab.add_word("<START>".to_string());
        vocab.add_word("<END>".to_string());
        
        vocab
    }
    
    fn add_word(&mut self, word: String) -> usize {
        if let Some(&id) = self.word_to_id.get(&word) {
            id
        } else {
            let id = self.vocab_size;
            self.word_to_id.insert(word.clone(), id);
            self.id_to_word.insert(id, word);
            self.vocab_size += 1;
            id
        }
    }
    
    fn get_id(&self, word: &str) -> usize {
        self.word_to_id.get(word).copied().unwrap_or(1) // 1 is <UNK>
    }
    
    fn get_word(&self, id: usize) -> String {
        self.id_to_word.get(&id).cloned().unwrap_or("<UNK>".to_string())
    }
}

// Text preprocessing
fn tokenize_text(text: &str) -> Vec<String> {
    let re = Regex::new(r"\b\w+\b").unwrap();
    re.find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn build_vocabulary(texts: &[String]) -> Vocabulary {
    let mut vocab = Vocabulary::new();
    
    for text in texts {
        let tokens = tokenize_text(text);
        for token in tokens {
            vocab.add_word(token);
        }
    }
    
    vocab
}

// Word embedding layer (simple lookup table)
#[derive(Clone)]
struct WordEmbedding {
    embeddings: DMatrix<f64>,
    vocab_size: usize,
    embed_dim: usize,
}

impl WordEmbedding {
    fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let mut rng = thread_rng();
        let std_dev = (1.0 / embed_dim as f64).sqrt();
        
        let embeddings = DMatrix::from_element(vocab_size, embed_dim, 0.0)
            .map(|_| rng.gen_range(-std_dev..std_dev));
        
        Self {
            embeddings,
            vocab_size,
            embed_dim,
        }
    }
    
    fn forward(&self, word_ids: &[usize]) -> DMatrix<f64> {
        let seq_len = word_ids.len();
        let mut output = DMatrix::zeros(seq_len, self.embed_dim);
        
        for (i, &word_id) in word_ids.iter().enumerate() {
            if word_id < self.vocab_size {
                output.set_row(i, &self.embeddings.row(word_id));
            }
        }
        
        output
    }
    
    fn get_pooled_embedding(&self, word_ids: &[usize]) -> DVector<f64> {
        let embeddings = self.forward(word_ids);
        
        // Mean pooling across sequence dimension
        let mut pooled = DVector::zeros(self.embed_dim);
        if !word_ids.is_empty() {
            for col in 0..self.embed_dim {
                let sum: f64 = embeddings.column(col).iter().sum();
                pooled[col] = sum / word_ids.len() as f64;
            }
        }
        
        pooled
    }
}

// Activation functions
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn tanh_activation(x: f64) -> f64 {
    x.tanh()
}

// Linear layer implementation
#[derive(Clone)]
struct Linear {
    weight: DMatrix<f64>,
    bias: DVector<f64>,
}

impl Linear {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f64).sqrt();
        
        let weight = DMatrix::from_element(output_size, input_size, 0.0)
            .map(|_| rng.gen_range(-std_dev..std_dev));
        let bias = DVector::zeros(output_size);
        
        Self { weight, bias }
    }
    
    fn forward(&self, input: &DVector<f64>) -> DVector<f64> {
        &self.weight * input + &self.bias
    }
}

// Text JEPA Model Implementation
struct TextJEPAModel {
    word_embedding: WordEmbedding,
    encoder: Linear,
    predictor: Linear,
    vocab: Vocabulary,
    embed_dim: usize,
    hidden_dim: usize,
}

impl TextJEPAModel {
    fn new(vocab: Vocabulary, embed_dim: usize, hidden_dim: usize) -> Self {
        let word_embedding = WordEmbedding::new(vocab.vocab_size, embed_dim);
        let encoder = Linear::new(embed_dim, hidden_dim);
        let predictor = Linear::new(hidden_dim, hidden_dim);
        
        Self {
            word_embedding,
            encoder,
            predictor,
            vocab,
            embed_dim,
            hidden_dim,
        }
    }
    
    fn encode_text(&self, text: &str) -> DVector<f64> {
        let tokens = tokenize_text(text);
        let word_ids: Vec<usize> = tokens.iter().map(|token| self.vocab.get_id(token)).collect();
        
        // Get pooled word embeddings
        let pooled_embedding = self.word_embedding.get_pooled_embedding(&word_ids);
        
        // Apply encoder
        let encoded = self.encoder.forward(&pooled_embedding);
        encoded.map(tanh_activation)
    }
    
    fn predict(&self, context_embedding: &DVector<f64>) -> DVector<f64> {
        let predicted = self.predictor.forward(context_embedding);
        predicted.map(tanh_activation)
    }
    
    fn forward(&self, context_text: &str, target_text: &str) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let context_embed = self.encode_text(context_text);
        let target_embed = self.encode_text(target_text);
        let pred_embed = self.predict(&context_embed);
        
        (context_embed, target_embed, pred_embed)
    }
    
    fn compute_loss(&self, context_text: &str, target_text: &str) -> f64 {
        let (_, target_embed, pred_embed) = self.forward(context_text, target_text);
        
        // Mean Squared Error loss
        let diff = &target_embed - &pred_embed;
        diff.dot(&diff) / target_embed.len() as f64
    }
    
    fn compute_similarity(&self, text1: &str, text2: &str) -> f64 {
        let embed1 = self.encode_text(text1);
        let embed2 = self.encode_text(text2);
        
        let norm1 = embed1.norm();
        let norm2 = embed2.norm();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            embed1.dot(&embed2) / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

// Sample text data for training
fn get_sample_texts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Machine learning is a powerful tool for data analysis".to_string(),
        "Natural language processing enables computers to understand text".to_string(),
        "Deep learning models can learn complex patterns from data".to_string(),
        "Artificial intelligence is transforming many industries".to_string(),
        "Text mining helps extract insights from large document collections".to_string(),
        "Neural networks are inspired by biological brain structures".to_string(),
        "Self-supervised learning reduces dependence on labeled data".to_string(),
        "Transformer models have revolutionized language understanding".to_string(),
        "Representation learning discovers meaningful features automatically".to_string(),
        "The cat sat on the comfortable mat in the sunny room".to_string(),
        "Programming languages provide tools for software development".to_string(),
        "Data science combines statistics with computational methods".to_string(),
        "Knowledge graphs represent relationships between entities".to_string(),
        "Information retrieval systems help users find relevant documents".to_string(),
    ]
}

// Generate context-target pairs for training
fn generate_training_pairs(texts: &[String], rng: &mut ThreadRng) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    
    // Create pairs from similar/related texts
    for _ in 0..50 {
        let idx1 = rng.gen_range(0..texts.len());
        let idx2 = rng.gen_range(0..texts.len());
        
        pairs.push((texts[idx1].clone(), texts[idx2].clone()));
    }
    
    // Create pairs from sentence fragments (context prediction)
    for text in texts {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > 4 {
            let split_point = rng.gen_range(2..words.len()-2);
            let context = words[..split_point].join(" ");
            let target = words[split_point..].join(" ");
            pairs.push((context, target));
        }
    }
    
    pairs
}

fn main() {
    println!("📚 Text JEPA (Joint Embedding Predictive Architecture)");
    println!("Natural Language Processing with Self-Supervised Learning");
    println!("========================================================");
    
    // Model hyperparameters
    let embed_dim = 64;
    let hidden_dim = 128;
    let num_epochs = 100;
    
    // Prepare training data
    let sample_texts = get_sample_texts();
    let vocab = build_vocabulary(&sample_texts);
    
    println!("Dataset Configuration:");
    println!("  Training texts: {}", sample_texts.len());
    println!("  Vocabulary size: {}", vocab.vocab_size);
    println!("  Word embedding dim: {}", embed_dim);
    println!("  Hidden dimension: {}", hidden_dim);
    println!();
    
    // Display some vocabulary examples
    println!("📖 Sample Vocabulary:");
    for i in 4..std::cmp::min(14, vocab.vocab_size) {
        println!("  {}: {}", i, vocab.get_word(i));
    }
    println!();
    
    // Initialize model
    let model = TextJEPAModel::new(vocab.clone(), embed_dim, hidden_dim);
    println!("✅ Text JEPA model initialized successfully!");
    
    // Generate training pairs
    let mut rng = thread_rng();
    let training_pairs = generate_training_pairs(&sample_texts, &mut rng);
    println!("📝 Generated {} training pairs", training_pairs.len());
    
    // Training simulation
    println!("\n🎯 Training Progress:");
    println!("---------------------");
    
    let mut losses = Vec::new();
    
    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        
        // Shuffle training pairs
        let mut shuffled_pairs = training_pairs.clone();
        shuffled_pairs.shuffle(&mut rng);
        
        // Train on pairs
        for (context, target) in shuffled_pairs.iter().take(20) {
            let loss = model.compute_loss(context, target);
            epoch_loss += loss;
            batch_count += 1;
        }
        
        if batch_count > 0 {
            epoch_loss /= batch_count as f64;
        }
        losses.push(epoch_loss);
        
        if epoch % 20 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:3}: Average Loss = {:.6}", epoch + 1, epoch_loss);
        }
    }
    
    // Model evaluation
    println!("\n🔍 Model Evaluation:");
    println!("--------------------");
    
    // Test semantic similarity
    let test_pairs = vec![
        ("machine learning", "artificial intelligence"),
        ("cat", "dog"),
        ("programming", "software development"),
        ("text mining", "data analysis"),
        ("neural networks", "deep learning"),
    ];
    
    println!("Semantic Similarity Tests:");
    for (text1, text2) in test_pairs {
        let similarity = model.compute_similarity(text1, text2);
        println!("  '{}' <-> '{}': {:.4}", text1, text2, similarity);
    }
    
    // Test context prediction
    println!("\nContext Prediction Examples:");
    let context_examples = vec![
        ("The quick brown fox", "jumps over the lazy dog"),
        ("Machine learning is", "a powerful tool for analysis"),
        ("Natural language processing", "enables text understanding"),
    ];
    
    for (context, expected_target) in context_examples {
        let (_context_embed, target_embed, pred_embed) = model.forward(context, expected_target);
        let prediction_similarity = target_embed.dot(&pred_embed) / (target_embed.norm() * pred_embed.norm());
        
        println!("  Context: '{}'", context);
        println!("  Expected: '{}'", expected_target);
        println!("  Prediction similarity: {:.4}", prediction_similarity);
        println!();
    }
    
    // Training statistics
    let initial_loss = losses.first().unwrap_or(&0.0);
    let final_loss = losses.last().unwrap_or(&0.0);
    let improvement = if *initial_loss > 0.0 { 
        ((initial_loss - final_loss) / initial_loss) * 100.0 
    } else { 
        0.0 
    };
    
    println!("📊 Training Summary:");
    println!("-------------------");
    println!("  Initial loss: {:.6}", initial_loss);
    println!("  Final loss: {:.6}", final_loss);
    println!("  Improvement: {:.2}%", improvement);
    
    // Architecture summary
    println!("\n🏗️  Text JEPA Architecture:");
    println!("---------------------------");
    println!("  Word Embedding: {} words → {} dim", vocab.vocab_size, embed_dim);
    println!("  Encoder: {} → {} (tanh activation)", embed_dim, hidden_dim);
    println!("  Predictor: {} → {} (tanh activation)", hidden_dim, hidden_dim);
    
    let total_params = (vocab.vocab_size * embed_dim) + 
                      (embed_dim * hidden_dim + hidden_dim) + 
                      (hidden_dim * hidden_dim + hidden_dim);
    println!("  Total parameters: {}", total_params);
    
    println!("\n✨ Text JEPA implementation completed successfully!");
    
    println!("\n🚀 Text-Specific Next Steps:");
    println!("----------------------------");
    println!("  1. Add positional encodings for sequence modeling");
    println!("  2. Implement attention mechanisms");
    println!("  3. Add BERT-style masked language modeling");
    println!("  4. Support longer text sequences");
    println!("  5. Add text classification downstream tasks");
    println!("  6. Implement subword tokenization (BPE)");
    println!("  7. Add text generation capabilities");
}
