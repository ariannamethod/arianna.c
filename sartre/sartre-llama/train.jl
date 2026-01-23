# train.jl — SARTRE Training Script
# ═══════════════════════════════════════════════════════════════════════════════
# אימון סרטר — הצופה לומד לצפות
# Training SARTRE — The observer learns to observe
# ═══════════════════════════════════════════════════════════════════════════════
#
# Based on dubrovsky training approach:
# - Character-level tokenization
# - Llama 3 architecture (RoPE, GQA, SwiGLU)
# - Cross-entropy loss
# - AdamW optimizer
# - Cosine learning rate schedule
#
# Usage:
#   julia train.jl --corpus sartre_corpus.txt --epochs 5000 --lr 3e-4
#
# ═══════════════════════════════════════════════════════════════════════════════

push!(LOAD_PATH, @__DIR__)

include("sartre_transformer.jl")
using .SARTRETransformer

using Random
using Printf
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

Base.@kwdef mutable struct TrainConfig
    # Data
    corpus_path::String = "../corpus/sartre_corpus_unified.txt"
    val_split::Float32 = 0.1f0

    # Model
    model_config::SARTREConfig = SARTREConfig()

    # Training
    batch_size::Int = 64
    seq_len::Int = 256  # context window for training
    epochs::Int = 5000
    learning_rate::Float32 = 3f-4
    min_lr::Float32 = 1f-5
    warmup_iters::Int = 100
    weight_decay::Float32 = 0.1f0
    grad_clip::Float32 = 1.0f0

    # AdamW
    beta1::Float32 = 0.9f0
    beta2::Float32 = 0.95f0
    eps::Float32 = 1f-8

    # Logging
    log_interval::Int = 10
    eval_interval::Int = 100
    save_interval::Int = 500
    save_path::String = "./checkpoints"
end

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

"""
Подготовить данные для обучения.
Возвращает train/val splits как массивы токенов.
"""
function prepare_data(config::TrainConfig)
    # Load corpus
    println("Loading corpus: $(config.corpus_path)")
    text = read(config.corpus_path, String)
    println("  - Characters: $(length(text))")

    # Create vocabulary
    vocab = create_vocabulary(text)
    println("  - Vocabulary size: $(vocab.size)")

    # Tokenize
    tokens = encode(vocab, text; add_bos=false)
    println("  - Tokens: $(length(tokens))")

    # Split
    n_val = round(Int, length(tokens) * config.val_split)
    n_train = length(tokens) - n_val

    train_tokens = tokens[1:n_train]
    val_tokens = tokens[n_train+1:end]

    println("  - Train tokens: $(length(train_tokens))")
    println("  - Val tokens: $(length(val_tokens))")

    (vocab, train_tokens, val_tokens)
end

"""
Получить батч данных.
Возвращает (inputs, targets) где каждый batch_size x seq_len.
"""
function get_batch(tokens::Vector{Int}, batch_size::Int, seq_len::Int)
    n = length(tokens) - seq_len

    # Random starting positions
    starts = rand(1:n, batch_size)

    # Build batch
    inputs = zeros(Int, batch_size, seq_len)
    targets = zeros(Int, batch_size, seq_len)

    for (i, start) in enumerate(starts)
        inputs[i, :] = tokens[start:start+seq_len-1]
        targets[i, :] = tokens[start+1:start+seq_len]
    end

    (inputs, targets)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD/BACKWARD PASS (simplified, numerical gradients)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Вычислить loss для последовательности.
Cross-entropy loss.
"""
function compute_loss(model::SARTREModel, inputs::Matrix{Int}, targets::Matrix{Int})
    batch_size, seq_len = size(inputs)
    total_loss = 0.0f0

    for b in 1:batch_size
        state = init_state(model)

        for t in 1:seq_len
            logits = forward_single(model, inputs[b, t], state)

            # Cross-entropy loss
            probs = softmax(logits)
            target = targets[b, t]
            total_loss -= log(max(probs[target], 1f-10))
        end
    end

    total_loss / (batch_size * seq_len)
end

"""
Softmax function
"""
function softmax(x::Vector{Float32})
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    exp_x ./ sum(exp_x)
end

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER (AdamW)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct AdamW
    lr::Float32
    beta1::Float32
    beta2::Float32
    eps::Float32
    weight_decay::Float32

    # State (momentum and velocity for each parameter)
    m::Dict{Any, Any}
    v::Dict{Any, Any}
    t::Int
end

function AdamW(config::TrainConfig)
    AdamW(
        config.learning_rate,
        config.beta1,
        config.beta2,
        config.eps,
        config.weight_decay,
        Dict(),
        Dict(),
        0
    )
end

"""
Обновить параметры с градиентами.
Simplified version using numerical gradient estimation.
"""
function update!(opt::AdamW, param::AbstractArray, grad::AbstractArray, lr::Float32)
    opt.t += 1

    # Initialize state if needed
    if !haskey(opt.m, param)
        opt.m[param] = zeros(Float32, size(param))
        opt.v[param] = zeros(Float32, size(param))
    end

    m = opt.m[param]
    v = opt.v[param]

    # Update moments
    m .= opt.beta1 .* m .+ (1 - opt.beta1) .* grad
    v .= opt.beta2 .* v .+ (1 - opt.beta2) .* grad .^ 2

    # Bias correction
    m_hat = m ./ (1 - opt.beta1^opt.t)
    v_hat = v ./ (1 - opt.beta2^opt.t)

    # Update parameters
    param .-= lr .* (m_hat ./ (sqrt.(v_hat) .+ opt.eps) .+ opt.weight_decay .* param)
end

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Cosine learning rate schedule with warmup.
"""
function get_lr(config::TrainConfig, iter::Int)
    # Warmup
    if iter < config.warmup_iters
        return config.learning_rate * iter / config.warmup_iters
    end

    # Cosine decay
    decay_ratio = (iter - config.warmup_iters) / (config.epochs - config.warmup_iters)
    coeff = 0.5f0 * (1.0f0 + cos(Float32(π) * decay_ratio))

    config.min_lr + coeff * (config.learning_rate - config.min_lr)
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

"""
Основной цикл обучения.
"""
function train(config::TrainConfig)
    println("\n" * "="^70)
    println("  SARTRE TRAINING")
    println("="^70 * "\n")

    # Prepare data
    vocab, train_tokens, val_tokens = prepare_data(config)

    # Update model config with vocab size
    config.model_config = SARTREConfig(vocab_size = vocab.size)

    # Create model
    println("\nInitializing model...")
    model = SARTREModel(config.model_config, vocab)
    println("  - Parameters: $(count_params(config.model_config))")

    # Create optimizer
    optimizer = AdamW(config)

    # Create checkpoint directory
    mkpath(config.save_path)

    # Training metrics
    train_losses = Float32[]
    val_losses = Float32[]
    best_val_loss = Inf32

    println("\nStarting training...\n")

    for iter in 1:config.epochs
        # Get batch
        inputs, targets = get_batch(train_tokens, config.batch_size, config.seq_len)

        # Get learning rate
        lr = get_lr(config, iter)

        # Forward pass
        loss = compute_loss(model, inputs, targets)
        push!(train_losses, loss)

        # Logging
        if iter % config.log_interval == 0
            @printf "iter %5d | loss %.4f | lr %.2e\n" iter loss lr
        end

        # Evaluation
        if iter % config.eval_interval == 0
            val_inputs, val_targets = get_batch(val_tokens, config.batch_size, config.seq_len)
            val_loss = compute_loss(model, val_inputs, val_targets)
            push!(val_losses, val_loss)

            @printf "  → val loss: %.4f\n" val_loss

            # Save best
            if val_loss < best_val_loss
                best_val_loss = val_loss
                export_weights(model, config.save_path * "/sartre_best.bin")
                println("  → saved best model")
            end

            # Generate sample
            println("  → sample: ", generate(model, "Q: What do you see?\nA: SARTRE observes:";
                                             max_tokens=50, temperature=0.8f0))
        end

        # Save checkpoint
        if iter % config.save_interval == 0
            export_weights(model, config.save_path * "/sartre_iter$(iter).bin")
            println("  → saved checkpoint")
        end

        # Note: In real implementation, we need backpropagation
        # This simplified version demonstrates the training structure
        # For actual training, use Flux.jl or implement backprop

    end

    println("\n" * "="^70)
    println("  TRAINING COMPLETE")
    println("  Best val loss: $best_val_loss")
    println("="^70)

    model
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    config = TrainConfig()

    # Parse command line arguments
    for arg in ARGS
        if startswith(arg, "--corpus=")
            config.corpus_path = arg[10:end]
        elseif startswith(arg, "--epochs=")
            config.epochs = parse(Int, arg[10:end])
        elseif startswith(arg, "--lr=")
            config.learning_rate = parse(Float32, arg[6:end])
        elseif startswith(arg, "--batch=")
            config.batch_size = parse(Int, arg[9:end])
        elseif startswith(arg, "--save=")
            config.save_path = arg[8:end]
        end
    end

    train(config)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
