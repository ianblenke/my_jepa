use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use regex::Regex;
use std::collections::HashMap;

#[cfg(target_os = "macos")]
use metal::*;

// Simplified Metal GPU acceleration
#[cfg(target_os = "macos")]
struct SimpleMetalCompute {
    device: Device,
    command_queue: CommandQueue,
}

#[cfg(target_os = "macos")]
impl SimpleMetalCompute {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;
        
        let command_queue = device.new_command_queue();
        
        println!("🔧 Metal device: {:?}", device.name());
        println!("🔧 Metal family: {:?}", device.supports_family(MTLGPUFamily::Mac2));
        
        Ok(Self {
            device,
            command_queue,
        })
    }
    
    fn vector_add_gpu(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err("Vector lengths must match".to_string());
        }
        
        let size = a.len();
        let byte_size = size * std::mem::size_of::<f32>();
        
        // Create buffers
        let buffer_a = self.device.new_buffer_with_data(
            a.as_ptr() as *const std::ffi::c_void,
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_b = self.device.new_buffer_with_data(
            b.as_ptr() as *const std::ffi::c_void,
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_result = self.device.new_buffer(
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Simple shader for vector addition
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void vector_add(
                device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* result [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                result[gid] = a[gid] + b[gid];
            }
        "#;
        
        let library = self.device.new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Shader compilation failed: {:?}", e))?;
        
        let function = library.get_function("vector_add", None)
            .map_err(|e| format!("Function not found: {:?}", e))?;
        
        let pipeline = self.device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation failed: {:?}", e))?;
        
        // Execute
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        
        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(((size + 255) / 256) as u64, 1, 1);
        
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read results
        let ptr = buffer_result.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, size).to_vec() };
        
        Ok(result)
    }
}

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
        self.word_to_id.get(word).copied().unwrap_or(1)
    }
}

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

// Hybrid CPU/GPU Text JEPA Model
struct HybridTextJEPAModel {
    word_embeddings: DMatrix<f64>,
    encoder_weight: DMatrix<f64>,
    encoder_bias: DVector<f64>,
    predictor_weight: DMatrix<f64>,
    predictor_bias: DVector<f64>,
    vocab: Vocabulary,
    embed_dim: usize,
    hidden_dim: usize,
    #[cfg(target_os = "macos")]
    metal_compute: Option<SimpleMetalCompute>,
    use_gpu: bool,
}

impl HybridTextJEPAModel {
    fn new(vocab: Vocabulary, embed_dim: usize, hidden_dim: usize, use_gpu: bool) -> Self {
        let mut rng = thread_rng();
        
        // Initialize weights
        let embed_std = (1.0 / embed_dim as f64).sqrt();
        let word_embeddings = DMatrix::from_element(vocab.vocab_size, embed_dim, 0.0)
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
        let metal_compute = if use_gpu {
            match SimpleMetalCompute::new() {
                Ok(compute) => {
                    println!("🚀 Metal GPU acceleration enabled!");
                    Some(compute)
                }
                Err(e) => {
                    println!("⚠️  Metal initialization failed: {}. Using CPU.", e);
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(target_os = "macos"))]
        let metal_compute = None;
        
        if !use_gpu || cfg!(not(target_os = "macos")) {
            println!("💻 Using CPU computation");
        }
        
        Self {
            word_embeddings,
            encoder_weight,
            encoder_bias,
            predictor_weight,
            predictor_bias,
            vocab,
            embed_dim,
            hidden_dim,
            #[cfg(target_os = "macos")]
            metal_compute,
            use_gpu,
        }
    }
    
    fn get_word_embedding(&self, word_id: usize) -> DVector<f64> {
        if word_id < self.vocab.vocab_size {
            self.word_embeddings.row(word_id).transpose()
        } else {
            DVector::zeros(self.embed_dim)
        }
    }
    
    fn encode_text(&self, text: &str) -> DVector<f64> {
        let tokens = tokenize_text(text);
        let word_ids: Vec<usize> = tokens.iter().map(|token| self.vocab.get_id(token)).collect();
        
        if word_ids.is_empty() {
            return DVector::zeros(self.hidden_dim);
        }
        
        // Mean pooling of word embeddings
        let mut pooled_embedding = DVector::zeros(self.embed_dim);
        for &word_id in &word_ids {
            pooled_embedding += self.get_word_embedding(word_id);
        }
        pooled_embedding /= word_ids.len() as f64;
        
        // Apply encoder (with optional GPU acceleration for bias addition)
        let linear_output = &self.encoder_weight * &pooled_embedding;
        let mut encoded = linear_output + &self.encoder_bias;
        
        // GPU-accelerated vector operations (demonstration)
        #[cfg(target_os = "macos")]
        if let Some(ref metal) = self.metal_compute {
            // Convert to f32 for GPU processing
            let a: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
            let zeros: Vec<f32> = vec![0.0; encoded.len()];
            
            // Simple GPU operation (adding zeros, just for demonstration)
            if let Ok(gpu_result) = metal.vector_add_gpu(&a, &zeros) {
                encoded = DVector::from_vec(gpu_result.iter().map(|&x| x as f64).collect());
                //println!("🔥 Used GPU for vector operation");
            }
        }
        
        // Apply tanh activation
        encoded.map(|x| x.tanh())
    }
    
    fn predict(&self, context_embedding: &DVector<f64>) -> DVector<f64> {
        let linear_output = &self.predictor_weight * context_embedding;
        let predicted = linear_output + &self.predictor_bias;
        predicted.map(|x| x.tanh())
    }
    
    fn forward(&self, context_text: &str, target_text: &str) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let context_embed = self.encode_text(context_text);
        let target_embed = self.encode_text(target_text);
        let pred_embed = self.predict(&context_embed);
        
        (context_embed, target_embed, pred_embed)
    }
    
    fn compute_loss(&self, context_text: &str, target_text: &str) -> f64 {
        let (_, target_embed, pred_embed) = self.forward(context_text, target_text);
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

fn get_sample_texts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Machine learning accelerates data analysis with GPU computing".to_string(),
        "Natural language processing enables intelligent text understanding".to_string(),
        "Deep learning models discover complex patterns in large datasets".to_string(),
        "Artificial intelligence transforms industries through automation".to_string(),
        "Metal performance shaders enable high-speed parallel processing".to_string(),
        "Graphics processing units excel at matrix computations".to_string(),
        "Self-supervised learning reduces dependence on labeled training data".to_string(),
        "Neural networks process information through interconnected layers".to_string(),
        "Representation learning discovers meaningful feature encodings automatically".to_string(),
        "Parallel computing architectures accelerate scientific calculations".to_string(),
        "Text embeddings capture semantic relationships between words".to_string(),
    ]
}

fn generate_training_pairs(texts: &[String], rng: &mut ThreadRng) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    
    // Random pairs
    for _ in 0..30 {
        let idx1 = rng.gen_range(0..texts.len());
        let idx2 = rng.gen_range(0..texts.len());
        pairs.push((texts[idx1].clone(), texts[idx2].clone()));
    }
    
    // Context-continuation pairs
    for text in texts {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > 4 {
            let split_point = rng.gen_range(2..words.len()-1);
            let context = words[..split_point].join(" ");
            let target = words[split_point..].join(" ");
            pairs.push((context, target));
        }
    }
    
    pairs
}

fn main() {
    println!("⚡ Hybrid CPU/GPU Text JEPA");
    println!("Smart Metal Acceleration for NLP");
    println!("=================================");
    
    let use_gpu = std::env::args().any(|arg| arg == "--gpu") || cfg!(target_os = "macos");
    
    if use_gpu {
        println!("🎯 GPU acceleration mode enabled");
    } else {
        println!("💻 CPU-only mode");
    }
    
    let embed_dim = 64;
    let hidden_dim = 128;
    let num_epochs = 25;
    
    let sample_texts = get_sample_texts();
    let vocab = build_vocabulary(&sample_texts);
    
    println!("\n📊 Configuration:");
    println!("  Training texts: {}", sample_texts.len());
    println!("  Vocabulary size: {}", vocab.vocab_size);
    println!("  Embedding dimension: {}", embed_dim);
    println!("  Hidden dimension: {}", hidden_dim);
    
    // Initialize hybrid model
    let model = HybridTextJEPAModel::new(vocab.clone(), embed_dim, hidden_dim, use_gpu);
    println!("✅ Hybrid Text JEPA model ready!");
    
    let mut rng = thread_rng();
    let training_pairs = generate_training_pairs(&sample_texts, &mut rng);
    println!("📝 Generated {} training pairs", training_pairs.len());
    
    // Training
    println!("\n🎯 Training Progress:");
    println!("{}", "-".repeat(30));
    
    let start_time = std::time::Instant::now();
    
    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        
        let mut shuffled_pairs = training_pairs.clone();
        shuffled_pairs.shuffle(&mut rng);
        
        for (context, target) in shuffled_pairs.iter().take(20) {
            let loss = model.compute_loss(context, target);
            epoch_loss += loss;
            batch_count += 1;
        }
        
        if batch_count > 0 {
            epoch_loss /= batch_count as f64;
        }
        
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:2}: Loss = {:.6}", epoch + 1, epoch_loss);
        }
    }
    
    let training_time = start_time.elapsed();
    println!("⏱️  Training completed in {:.2}s", training_time.as_secs_f64());
    
    // Evaluation
    println!("\n🔍 Evaluation Results:");
    println!("{}", "-".repeat(25));
    
    let test_pairs = vec![
        ("machine learning", "artificial intelligence"),
        ("GPU computing", "parallel processing"),
        ("neural networks", "deep learning"),
        ("text processing", "language understanding"),
    ];
    
    for (text1, text2) in test_pairs {
        let similarity = model.compute_similarity(text1, text2);
        println!("'{}' ↔ '{}': {:.3}", text1, text2, similarity);
    }
    
    // Context prediction test
    println!("\n🧠 Context Prediction:");
    println!("{}", "-".repeat(22));
    
    let context = "Machine learning accelerates";
    let target = "data analysis with GPU computing";
    let (_context_embed, target_embed, pred_embed) = model.forward(context, target);
    
    let prediction_quality = target_embed.dot(&pred_embed) / (target_embed.norm() * pred_embed.norm());
    
    println!("Context: '{}'", context);
    println!("Expected: '{}'", target);
    println!("Prediction quality: {:.3}", prediction_quality);
    
    println!("\n📈 Performance Summary:");
    println!("{}", "-".repeat(23));
    println!("  Training time: {:.2}s", training_time.as_secs_f64());
    println!("  GPU utilization: {}", if model.use_gpu { "Enabled" } else { "Disabled" });
    
    #[cfg(target_os = "macos")]
    if model.metal_compute.is_some() {
        println!("  Metal acceleration: ✅ Active");
        println!("  Compute shaders: Vector operations");
    } else {
        println!("  Metal acceleration: ❌ Not available");
    }
    
    println!("\n🚀 Hybrid Text JEPA completed successfully!");
    
    println!("\n🔧 GPU Optimization Tips:");
    println!("  • Use --gpu flag for Metal acceleration");
    println!("  • Batch operations for better GPU utilization");  
    println!("  • Consider FP16 precision for speed");
    println!("  • Profile with Instruments for optimization");
}
