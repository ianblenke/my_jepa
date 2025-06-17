use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use clap::{Parser, Subcommand};
use tokio;

#[cfg(target_os = "macos")]
use metal::*;

// Command line interface
#[derive(Parser)]
#[command(name = "jepa")]
#[command(about = "Self-Learning JEPA Text Model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model on text data
    Train {
        /// Input text file or directory
        #[arg(short, long)]
        input: String,
        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,
        /// Use GPU acceleration
        #[arg(long)]
        gpu: bool,
        /// Learning rate
        #[arg(short, long, default_value = "0.001")]
        lr: f64,
    },
    /// Continue training from existing model
    Continue {
        /// Model file to load
        #[arg(short, long)]
        model: String,
        /// Additional training data
        #[arg(short, long)]
        input: String,
        /// Number of additional epochs
        #[arg(short, long, default_value = "50")]
        epochs: usize,
    },
    /// Evaluate model on text
    Evaluate {
        /// Model file to load
        #[arg(short, long)]
        model: String,
        /// Text to evaluate
        #[arg(short, long)]
        text: String,
    },
    /// Interactive mode for continuous learning
    Interactive {
        /// Model file to load/save
        #[arg(short, long, default_value = "model.json")]
        model: String,
        /// Auto-save interval in minutes
        #[arg(long, default_value = "10")]
        autosave: u64,
    },
    /// Serve model as API
    Serve {
        /// Model file to load
        #[arg(short, long)]
        model: String,
        /// Port to serve on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}

// Serializable model state
#[derive(Serialize, Deserialize, Clone)]
struct ModelState {
    word_embeddings: Vec<Vec<f64>>,
    encoder_weight: Vec<Vec<f64>>,
    encoder_bias: Vec<f64>,
    predictor_weight: Vec<Vec<f64>>,
    predictor_bias: Vec<f64>,
    vocab_word_to_id: HashMap<String, usize>,
    vocab_id_to_word: HashMap<usize, String>,
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    training_stats: TrainingStats,
}

#[derive(Serialize, Deserialize, Clone)]
struct TrainingStats {
    total_epochs: usize,
    total_samples: usize,
    average_loss: f64,
    last_updated: String,
    learning_rate: f64,
}

// Vocabulary with frequency tracking
#[derive(Debug, Clone)]
struct AdaptiveVocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    word_frequency: HashMap<String, usize>,
    vocab_size: usize,
    min_frequency: usize,
}

impl AdaptiveVocabulary {
    fn new(min_frequency: usize) -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            word_frequency: HashMap::new(),
            vocab_size: 0,
            min_frequency,
        };
        
        // Add special tokens
        vocab.add_word("<PAD>".to_string(), true);
        vocab.add_word("<UNK>".to_string(), true);
        vocab.add_word("<START>".to_string(), true);
        vocab.add_word("<END>".to_string(), true);
        vocab.add_word("<MASK>".to_string(), true);
        
        vocab
    }
    
    fn add_word(&mut self, word: String, force: bool) -> usize {
        // Update frequency
        *self.word_frequency.entry(word.clone()).or_insert(0) += 1;
        
        // Only add to vocab if frequency threshold is met or forced
        if force || self.word_frequency[&word] >= self.min_frequency {
            if let Some(&id) = self.word_to_id.get(&word) {
                id
            } else {
                let id = self.vocab_size;
                self.word_to_id.insert(word.clone(), id);
                self.id_to_word.insert(id, word);
                self.vocab_size += 1;
                id
            }
        } else {
            1 // Return <UNK> token
        }
    }
    
    fn get_id(&self, word: &str) -> usize {
        self.word_to_id.get(word).copied().unwrap_or(1) // <UNK>
    }
    
    fn prune_rare_words(&mut self) {
        // Remove words below frequency threshold
        let words_to_remove: Vec<String> = self.word_frequency
            .iter()
            .filter(|(_, &freq)| freq < self.min_frequency)
            .map(|(word, _)| word.clone())
            .collect();
        
        for word in words_to_remove {
            if let Some(id) = self.word_to_id.remove(&word) {
                self.id_to_word.remove(&id);
                self.word_frequency.remove(&word);
            }
        }
        
        // Rebuild vocab with new IDs
        self.rebuild_vocab();
    }
    
    fn rebuild_vocab(&mut self) {
        let mut words: Vec<String> = self.word_to_id.keys().cloned().collect();
        words.sort();
        
        self.word_to_id.clear();
        self.id_to_word.clear();
        self.vocab_size = 0;
        
        for word in words {
            let id = self.vocab_size;
            self.word_to_id.insert(word.clone(), id);
            self.id_to_word.insert(id, word);
            self.vocab_size += 1;
        }
    }
}

// Advanced tokenizer with subword support
struct AdvancedTokenizer {
    word_regex: Regex,
    subword_regex: Regex,
    use_subwords: bool,
}

impl AdvancedTokenizer {
    fn new(use_subwords: bool) -> Self {
        Self {
            word_regex: Regex::new(r"\b\w+\b").unwrap(),
            subword_regex: Regex::new(r"\w{2,}").unwrap(),
            use_subwords,
        }
    }
    
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        
        // Extract words
        for word_match in self.word_regex.find_iter(text) {
            let word = word_match.as_str().to_lowercase();
            
            if self.use_subwords && word.len() > 4 {
                // Add subword tokens for longer words
                tokens.push(format!("▁{}", word)); // Whole word
                
                // Add character n-grams
                for i in 0..word.len().saturating_sub(2) {
                    tokens.push(word[i..i+3].to_string());
                }
            } else {
                tokens.push(word);
            }
        }
        
        tokens
    }
}

// GPU-accelerated gradient computation
#[cfg(target_os = "macos")]
struct MetalGradientCompute {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
}

#[cfg(target_os = "macos")]
impl MetalGradientCompute {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;
        
        let command_queue = device.new_command_queue();
        
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void compute_gradients(
                device const float* predictions [[buffer(0)]],
                device const float* targets [[buffer(1)]],
                device float* gradients [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                gradients[gid] = 2.0 * (predictions[gid] - targets[gid]) / float(size);
            }
            
            kernel void apply_gradients(
                device float* weights [[buffer(0)]],
                device const float* gradients [[buffer(1)]],
                constant float& learning_rate [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                weights[gid] -= learning_rate * gradients[gid];
            }
        "#;
        
        let library = device.new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Shader compilation failed: {:?}", e))?;
        
        Ok(Self {
            device,
            command_queue,
            library,
        })
    }
    
    fn compute_and_apply_gradients(&self, predictions: &mut [f32], targets: &[f32], 
                                   weights: &mut [f32], learning_rate: f32) -> Result<(), String> {
        let size = predictions.len();
        
        // Create buffers
        let pred_buffer = self.device.new_buffer_with_data(
            predictions.as_ptr() as *const std::ffi::c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let target_buffer = self.device.new_buffer_with_data(
            targets.as_ptr() as *const std::ffi::c_void,
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let grad_buffer = self.device.new_buffer(
            (size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let weight_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Compute gradients
        let grad_function = self.library.get_function("compute_gradients", None)
            .map_err(|e| format!("Gradient function not found: {:?}", e))?;
        let grad_pipeline = self.device.new_compute_pipeline_state_with_function(&grad_function)
            .map_err(|e| format!("Gradient pipeline creation failed: {:?}", e))?;
        
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&grad_pipeline);
        encoder.set_buffer(0, Some(&pred_buffer), 0);
        encoder.set_buffer(1, Some(&target_buffer), 0);
        encoder.set_buffer(2, Some(&grad_buffer), 0);
        
        let size_buffer = self.device.new_buffer_with_data(
            &(size as u32) as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(3, Some(&size_buffer), 0);
        
        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(((size + 255) / 256) as u64, 1, 1);
        
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Apply gradients
        let apply_function = self.library.get_function("apply_gradients", None)
            .map_err(|e| format!("Apply function not found: {:?}", e))?;
        let apply_pipeline = self.device.new_compute_pipeline_state_with_function(&apply_function)
            .map_err(|e| format!("Apply pipeline creation failed: {:?}", e))?;
        
        let command_buffer2 = self.command_queue.new_command_buffer();
        let encoder2 = command_buffer2.new_compute_command_encoder();
        
        encoder2.set_compute_pipeline_state(&apply_pipeline);
        encoder2.set_buffer(0, Some(&weight_buffer), 0);
        encoder2.set_buffer(1, Some(&grad_buffer), 0);
        
        let lr_buffer = self.device.new_buffer_with_data(
            &learning_rate as *const f32 as *const std::ffi::c_void,
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder2.set_buffer(2, Some(&lr_buffer), 0);
        
        let weight_size_buffer = self.device.new_buffer_with_data(
            &(weights.len() as u32) as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder2.set_buffer(3, Some(&weight_size_buffer), 0);
        
        let thread_groups2 = MTLSize::new(((weights.len() + 255) / 256) as u64, 1, 1);
        encoder2.dispatch_thread_groups(thread_groups2, thread_group_size);
        encoder2.end_encoding();
        
        command_buffer2.commit();
        command_buffer2.wait_until_completed();
        
        // Copy results back
        let weight_ptr = weight_buffer.contents() as *const f32;
        let updated_weights = unsafe { std::slice::from_raw_parts(weight_ptr, weights.len()) };
        weights.copy_from_slice(updated_weights);
        
        Ok(())
    }
}

// Self-learning JEPA model with continuous adaptation
struct SelfLearningJEPA {
    word_embeddings: DMatrix<f64>,
    encoder_weight: DMatrix<f64>,
    encoder_bias: DVector<f64>,
    predictor_weight: DMatrix<f64>,
    predictor_bias: DVector<f64>,
    vocab: AdaptiveVocabulary,
    tokenizer: AdvancedTokenizer,
    embed_dim: usize,
    hidden_dim: usize,
    learning_rate: f64,
    stats: TrainingStats,
    #[cfg(target_os = "macos")]
    gpu_compute: Option<MetalGradientCompute>,
    experience_buffer: Vec<(String, String)>, // Store recent training examples
    max_buffer_size: usize,
}

impl SelfLearningJEPA {
    fn new(embed_dim: usize, hidden_dim: usize, learning_rate: f64, use_gpu: bool) -> Self {
        let mut vocab = AdaptiveVocabulary::new(2); // Minimum frequency of 2
        let tokenizer = AdvancedTokenizer::new(true); // Use subwords
        
        // Initialize with proper dimensions after vocab is set up
        let mut rng = thread_rng();
        
        // Start with minimal embeddings, will grow as vocab grows
        let initial_vocab_size = std::cmp::max(vocab.vocab_size, 10);
        let embed_std = (1.0 / embed_dim as f64).sqrt();
        let word_embeddings = DMatrix::from_element(initial_vocab_size, embed_dim, 0.0)
            .map(|_| rng.gen_range(-embed_std..embed_std));
        
        let encoder_std = (2.0 / (embed_dim + hidden_dim) as f64).sqrt();
        let encoder_weight = DMatrix::from_element(hidden_dim, embed_dim, 0.0)
            .map(|_| rng.gen_range(-encoder_std..encoder_std));
        let encoder_bias = DVector::zeros(hidden_dim);
        
        let predictor_std = (2.0 / (hidden_dim * 2) as f64).sqrt();
        let predictor_weight = DMatrix::from_element(hidden_dim, hidden_dim, 0.0)
            .map(|_| rng.gen_range(-predictor_std..predictor_std));
        let predictor_bias = DVector::zeros(hidden_dim);
        
        #[cfg(target_os = "macos")]
        let gpu_compute = if use_gpu {
            MetalGradientCompute::new().ok()
        } else {
            None
        };
        
        let stats = TrainingStats {
            total_epochs: 0,
            total_samples: 0,
            average_loss: 0.0,
            last_updated: chrono::Utc::now().to_rfc3339(),
            learning_rate,
        };
        
        Self {
            word_embeddings,
            encoder_weight,
            encoder_bias,
            predictor_weight,
            predictor_bias,
            vocab,
            tokenizer,
            embed_dim,
            hidden_dim,
            learning_rate,
            stats,
            #[cfg(target_os = "macos")]
            gpu_compute,
            experience_buffer: Vec::new(),
            max_buffer_size: 10000,
        }
    }
    
    fn save_model(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let state = ModelState {
            word_embeddings: self.word_embeddings.as_slice().chunks(self.embed_dim)
                .map(|chunk| chunk.to_vec()).collect(),
            encoder_weight: self.encoder_weight.as_slice().chunks(self.embed_dim)
                .map(|chunk| chunk.to_vec()).collect(),
            encoder_bias: self.encoder_bias.as_slice().to_vec(),
            predictor_weight: self.predictor_weight.as_slice().chunks(self.hidden_dim)
                .map(|chunk| chunk.to_vec()).collect(),
            predictor_bias: self.predictor_bias.as_slice().to_vec(),
            vocab_word_to_id: self.vocab.word_to_id.clone(),
            vocab_id_to_word: self.vocab.id_to_word.clone(),
            vocab_size: self.vocab.vocab_size,
            embed_dim: self.embed_dim,
            hidden_dim: self.hidden_dim,
            training_stats: self.stats.clone(),
        };
        
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(path, json)?;
        
        println!("💾 Model saved to {}", path);
        Ok(())
    }
    
    fn load_model(path: &str, use_gpu: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let state: ModelState = serde_json::from_str(&json)?;
        
        // Reconstruct matrices
        let word_embeddings = DMatrix::from_vec(state.vocab_size, state.embed_dim,
            state.word_embeddings.into_iter().flatten().collect());
        
        let encoder_weight = DMatrix::from_vec(state.hidden_dim, state.embed_dim,
            state.encoder_weight.into_iter().flatten().collect());
        let encoder_bias = DVector::from_vec(state.encoder_bias);
        
        let predictor_weight = DMatrix::from_vec(state.hidden_dim, state.hidden_dim,
            state.predictor_weight.into_iter().flatten().collect());
        let predictor_bias = DVector::from_vec(state.predictor_bias);
        
        // Reconstruct vocabulary
        let mut vocab = AdaptiveVocabulary::new(2);
        vocab.word_to_id = state.vocab_word_to_id;
        vocab.id_to_word = state.vocab_id_to_word;
        vocab.vocab_size = state.vocab_size;
        
        #[cfg(target_os = "macos")]
        let gpu_compute = if use_gpu {
            MetalGradientCompute::new().ok()
        } else {
            None
        };
        
        println!("📁 Model loaded from {}", path);
        
        Ok(Self {
            word_embeddings,
            encoder_weight,
            encoder_bias,
            predictor_weight,
            predictor_bias,
            vocab,
            tokenizer: AdvancedTokenizer::new(true),
            embed_dim: state.embed_dim,
            hidden_dim: state.hidden_dim,
            learning_rate: state.training_stats.learning_rate,
            stats: state.training_stats,
            #[cfg(target_os = "macos")]
            gpu_compute,
            experience_buffer: Vec::new(),
            max_buffer_size: 10000,
        })
    }
    
    fn encode_text(&self, text: &str) -> DVector<f64> {
        let tokens = self.tokenizer.tokenize(text);
        let word_ids: Vec<usize> = tokens.iter().map(|token| self.vocab.get_id(token)).collect();
        
        if word_ids.is_empty() {
            return DVector::zeros(self.hidden_dim);
        }
        
        // Mean pooling of word embeddings
        let mut pooled_embedding = DVector::zeros(self.embed_dim);
        let mut valid_embeddings = 0;
        
        for &word_id in &word_ids {
            if word_id < self.word_embeddings.nrows() {
                pooled_embedding += self.word_embeddings.row(word_id).transpose();
                valid_embeddings += 1;
            }
        }
        
        if valid_embeddings > 0 {
            pooled_embedding /= valid_embeddings as f64;
        }
        
        // Apply encoder
        let encoded = &self.encoder_weight * &pooled_embedding + &self.encoder_bias;
        encoded.map(|x| x.tanh())
    }
    
    fn predict(&self, context_embedding: &DVector<f64>) -> DVector<f64> {
        let predicted = &self.predictor_weight * context_embedding + &self.predictor_bias;
        predicted.map(|x| x.tanh())
    }
    
    fn train_online(&mut self, context_text: &str, target_text: &str) -> f64 {
        // Update vocabulary with new words
        let context_tokens = self.tokenizer.tokenize(context_text);
        let target_tokens = self.tokenizer.tokenize(target_text);
        
        let mut vocab_updated = false;
        let old_vocab_size = self.vocab.vocab_size;
        
        for token in context_tokens.iter().chain(target_tokens.iter()) {
            self.vocab.add_word(token.clone(), false);
        }
        
        // Resize embeddings if vocabulary grew
        if self.vocab.vocab_size > old_vocab_size {
            let new_rows = self.vocab.vocab_size - self.word_embeddings.nrows();
            if new_rows > 0 {
                let mut rng = thread_rng();
                let embed_std = (1.0 / self.embed_dim as f64).sqrt();
                
                // Create new embedding matrix with additional rows
                let mut new_embeddings = DMatrix::zeros(self.vocab.vocab_size, self.embed_dim);
                
                // Copy existing embeddings
                for i in 0..self.word_embeddings.nrows() {
                    for j in 0..self.word_embeddings.ncols() {
                        new_embeddings[(i, j)] = self.word_embeddings[(i, j)];
                    }
                }
                
                // Initialize new embeddings
                for i in self.word_embeddings.nrows()..self.vocab.vocab_size {
                    for j in 0..self.embed_dim {
                        new_embeddings[(i, j)] = rng.gen_range(-embed_std..embed_std);
                    }
                }
                
                self.word_embeddings = new_embeddings;
                vocab_updated = true;
            }
        }
        
        // Forward pass
        let context_embed = self.encode_text(context_text);
        let target_embed = self.encode_text(target_text);
        let pred_embed = self.predict(&context_embed);
        
        // Compute loss
        let diff = &target_embed - &pred_embed;
        let loss = diff.dot(&diff) / target_embed.len() as f64;
        
        // Simple gradient descent (CPU version)
        let grad_scale = 2.0 * self.learning_rate / target_embed.len() as f64;
        
        // Update predictor weights
        for i in 0..self.predictor_weight.nrows() {
            for j in 0..self.predictor_weight.ncols() {
                let gradient = diff[i] * context_embed[j] * grad_scale;
                self.predictor_weight[(i, j)] -= gradient;
            }
        }
        
        // Update predictor bias
        for i in 0..self.predictor_bias.len() {
            self.predictor_bias[i] -= diff[i] * grad_scale;
        }
        
        // Add to experience buffer
        self.experience_buffer.push((context_text.to_string(), target_text.to_string()));
        if self.experience_buffer.len() > self.max_buffer_size {
            self.experience_buffer.remove(0);
        }
        
        // Update stats
        self.stats.total_samples += 1;
        self.stats.average_loss = (self.stats.average_loss * 0.99) + (loss * 0.01);
        self.stats.last_updated = chrono::Utc::now().to_rfc3339();
        
        if vocab_updated {
            println!("📚 Vocabulary expanded to {} words", self.vocab.vocab_size);
        }
        
        loss
    }
    
    fn batch_train(&mut self, texts: &[String], epochs: usize) {
        println!("🎯 Training on {} texts for {} epochs", texts.len(), epochs);
        
        let mut rng = thread_rng();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;
            
            // Generate training pairs
            let mut training_pairs = Vec::new();
            
            // Context-continuation pairs
            for text in texts {
                let tokens = self.tokenizer.tokenize(text);
                if tokens.len() > 4 {
                    let split_point = rng.gen_range(2..tokens.len()-1);
                    let context = tokens[..split_point].join(" ");
                    let target = tokens[split_point..].join(" ");
                    training_pairs.push((context, target));
                }
            }
            
            // Sentence-sentence pairs
            for _ in 0..texts.len() {
                let idx1 = rng.gen_range(0..texts.len());
                let idx2 = rng.gen_range(0..texts.len());
                training_pairs.push((texts[idx1].clone(), texts[idx2].clone()));
            }
            
            // Train on pairs
            for (context, target) in training_pairs {
                let loss = self.train_online(&context, &target);
                epoch_loss += loss;
                batch_count += 1;
            }
            
            if batch_count > 0 {
                epoch_loss /= batch_count as f64;
            }
            
            if epoch % 20 == 0 || epoch == epochs - 1 {
                println!("Epoch {:3}: Loss = {:.6}", epoch + 1, epoch_loss);
            }
        }
        
        self.stats.total_epochs += epochs;
        
        // Prune rare words periodically
        if self.stats.total_epochs % 50 == 0 {
            println!("🧹 Pruning rare words...");
            self.vocab.prune_rare_words();
        }
    }
    
    fn interactive_mode(&mut self, model_path: &str, autosave_minutes: u64) {
        println!("🤖 Interactive JEPA Mode");
        println!("Type 'quit' to exit, 'save' to save model");
        println!("Enter text pairs separated by ' -> '");
        println!("Example: 'The cat sat' -> 'on the mat'");
        println!("Auto-save every {} minutes", autosave_minutes);
        
        let mut last_save = std::time::Instant::now();
        let autosave_duration = std::time::Duration::from_secs(autosave_minutes * 60);
        
        loop {
            println!("\n📝 Enter training text (or command):");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();
            
            if input == "quit" {
                break;
            } else if input == "save" {
                if let Err(e) = self.save_model(model_path) {
                    println!("❌ Error saving model: {}", e);
                }
                continue;
            } else if input == "stats" {
                println!("📊 Model Statistics:");
                println!("  Total epochs: {}", self.stats.total_epochs);
                println!("  Total samples: {}", self.stats.total_samples);
                println!("  Average loss: {:.6}", self.stats.average_loss);
                println!("  Vocabulary size: {}", self.vocab.vocab_size);
                println!("  Experience buffer: {}", self.experience_buffer.len());
                continue;
            }
            
            // Parse training pair
            if let Some(arrow_pos) = input.find(" -> ") {
                let context = input[..arrow_pos].trim();
                let target = input[arrow_pos + 4..].trim();
                
                let loss = self.train_online(context, target);
                println!("✅ Trained! Loss: {:.6}", loss);
                
                // Show similarity for feedback
                let similarity = self.compute_similarity(context, target);
                println!("📊 Similarity: {:.3}", similarity);
            } else {
                println!("❌ Invalid format. Use: 'context' -> 'target'");
            }
            
            // Auto-save check
            if last_save.elapsed() >= autosave_duration {
                println!("💾 Auto-saving model...");
                if let Err(e) = self.save_model(model_path) {
                    println!("❌ Auto-save failed: {}", e);
                }
                last_save = std::time::Instant::now();
            }
        }
        
        // Final save
        if let Err(e) = self.save_model(model_path) {
            println!("❌ Final save failed: {}", e);
        }
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

// Add chrono for timestamps
use chrono;

// Text loading utilities
fn load_text_files(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut texts = Vec::new();
    let path = Path::new(path);
    
    if path.is_file() {
        let content = fs::read_to_string(path)?;
        // Split into sentences
        let sentences: Vec<String> = content
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| s.trim().len() > 10)
            .map(|s| s.trim().to_string())
            .collect();
        texts.extend(sentences);
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.extension().map_or(false, |ext| ext == "txt") {
                let content = fs::read_to_string(file_path)?;
                let sentences: Vec<String> = content
                    .split(|c| c == '.' || c == '!' || c == '?')
                    .filter(|s| s.trim().len() > 10)
                    .map(|s| s.trim().to_string())
                    .collect();
                texts.extend(sentences);
            }
        }
    }
    
    Ok(texts)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Train { input, epochs, gpu, lr } => {
            println!("🚀 Training new JEPA model");
            
            let texts = load_text_files(&input)?;
            println!("📚 Loaded {} text segments", texts.len());
            
            let mut model = SelfLearningJEPA::new(64, 128, lr, gpu);
            model.batch_train(&texts, epochs);
            
            let model_path = "trained_model.json";
            model.save_model(model_path)?;
            
            println!("✅ Training complete! Model saved to {}", model_path);
        }
        
        Commands::Continue { model, input, epochs } => {
            println!("📈 Continuing training from existing model");
            
            let mut jepa_model = SelfLearningJEPA::load_model(&model, false)?;
            let texts = load_text_files(&input)?;
            
            println!("📚 Loaded {} additional text segments", texts.len());
            jepa_model.batch_train(&texts, epochs);
            
            jepa_model.save_model(&model)?;
            println!("✅ Continued training complete!");
        }
        
        Commands::Evaluate { model, text } => {
            println!("🔍 Evaluating model");
            
            let jepa_model = SelfLearningJEPA::load_model(&model, false)?;
            let embedding = jepa_model.encode_text(&text);
            
            println!("📊 Text: '{}'", text);
            println!("📊 Embedding norm: {:.4}", embedding.norm());
            println!("📊 Embedding dimension: {}", embedding.len());
        }
        
        Commands::Interactive { model, autosave } => {
            let mut jepa_model = if Path::new(&model).exists() {
                SelfLearningJEPA::load_model(&model, false)?
            } else {
                println!("🆕 Creating new model");
                SelfLearningJEPA::new(64, 128, 0.001, false)
            };
            
            jepa_model.interactive_mode(&model, autosave);
        }
        
        Commands::Serve { model, port } => {
            println!("🌐 Serving model on port {}", port);
            println!("API endpoints:");
            println!("  POST /encode - Encode text to embedding");
            println!("  POST /similarity - Compute text similarity");
            println!("  POST /train - Online training");
            
            // TODO: Implement web server
            println!("⚠️  Web server not implemented yet");
        }
    }
    
    Ok(())
}
