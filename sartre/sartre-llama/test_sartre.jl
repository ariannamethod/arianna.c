#!/usr/bin/env julia
# test_sartre.jl — Test SARTRE with trained weights
# ═══════════════════════════════════════════════════════════════════════════════

push!(LOAD_PATH, @__DIR__)

include("sartre_transformer.jl")
using .SARTRETransformer: SARTREConfig, SARTREModel, SARTREState, init_state, forward_single, sample_token, Vocabulary, encode, decode

using JSON3

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

function load_weights_from_bin(bin_path::String, config_path::String)
    """Load binary weights (float32)."""
    config_json = JSON3.read(read(config_path, String))

    dim = config_json.dim
    n_layers = config_json.n_layers
    n_heads = config_json.n_heads
    n_kv_heads = config_json.n_kv_heads
    vocab_size = config_json.vocab_size
    hidden_dim = config_json.hidden_dim

    head_dim = dim ÷ n_heads
    kv_dim = n_kv_heads * head_dim

    println("Loading SARTRE weights from binary...")
    println("  dim=$dim, layers=$n_layers, heads=$n_heads, vocab=$vocab_size")

    io = open(bin_path, "r")

    function read_matrix(rows::Int, cols::Int)
        data = Vector{Float32}(undef, rows * cols)
        read!(io, data)
        # Python exports row-major (rows x cols)
        # Julia expects column-major
        # Read as (rows x cols) row-major, convert to column-major
        mat = reshape(data, (cols, rows))
        transpose(mat) |> collect
    end

    function read_vector(len::Int)
        data = Vector{Float32}(undef, len)
        read!(io, data)
        data
    end

    cfg = SARTREConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=get(config_json, :max_seq_len, 512),
        ffn_hidden=hidden_dim
    )

    # Create dummy vocab for model init (will load real tokenizer later)
    dummy_vocab = Vocabulary(Dict{Char, Int}(), Dict{Int, Char}(), vocab_size)
    model = SARTREModel(cfg, dummy_vocab)

    # Token embeddings (transposed: Julia is dim x vocab)
    model.token_embed = read_matrix(vocab_size, dim)'

    # Layers
    for i in 1:n_layers
        println("  Layer $i...")
        model.layers[i].attn_norm.weight = read_vector(dim)
        model.layers[i].attention.wq = read_matrix(dim, dim)
        model.layers[i].attention.wk = read_matrix(kv_dim, dim)
        model.layers[i].attention.wv = read_matrix(kv_dim, dim)
        model.layers[i].attention.wo = read_matrix(dim, dim)
        model.layers[i].ffn_norm.weight = read_vector(dim)
        model.layers[i].feed_forward.w_gate = read_matrix(hidden_dim, dim)
        model.layers[i].feed_forward.w_up = read_matrix(hidden_dim, dim)
        model.layers[i].feed_forward.w_down = read_matrix(dim, hidden_dim)
    end

    # Final (output_proj is vocab x dim)
    model.final_norm.weight = read_vector(dim)
    model.output_proj = read_matrix(vocab_size, dim)

    close(io)

    println("✅ Weights loaded!")
    model
end

function load_tokenizer(tokenizer_path::String)
    """Load character-level tokenizer."""
    tok_json = JSON3.read(read(tokenizer_path, String))

    char_to_idx = Dict{Char, Int}()
    idx_to_char = Dict{Int, Char}()

    for (key, idx) in tok_json.char_to_id
        char_str = String(key)  # Convert Symbol/String to String
        char = first(char_str)  # Get first character
        char_to_idx[char] = idx + 1  # Julia 1-indexed
        idx_to_char[idx + 1] = char
    end

    vocab_size = tok_json.vocab_size
    println("Tokenizer loaded: vocab_size=$vocab_size")

    Vocabulary(char_to_idx, idx_to_char, vocab_size)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

function generate_text(
    model::SARTREModel,
    vocab::Vocabulary,
    prompt::String;
    max_tokens::Int=200,
    temperature::Float32=0.7f0,
    top_k::Int=40
)
    """Generate text from prompt."""
    tokens = encode(vocab, prompt; add_bos=false)
    state = init_state(model)

    println("\nPrompt: \"$prompt\"")
    print("SARTRE: ")

    generated = Char[]

    for step in 1:max_tokens
        # Forward
        logits = forward_single(model, tokens[end], state)

        # Temperature
        logits ./= temperature

        # Top-k
        if top_k > 0 && length(logits) > top_k
            sorted_indices = sortperm(logits, rev=true)
            threshold = logits[sorted_indices[top_k]]
            logits[logits .< threshold] .= -Inf32
        end

        # Sample
        next_token = sample_token(logits)
        push!(tokens, next_token)

        # Decode
        if haskey(vocab.idx_to_char, next_token)
            char = vocab.idx_to_char[next_token]
            push!(generated, char)
            print(char)

            # Stop at complete answer
            text = String(generated)
            if occursin("\nQ:", text) && length(text) > 20
                break
            end
            if char == '\n' && length(text) > 50 && !endswith(text, "A:\n")
                break
            end
        end

        flush(stdout)
    end

    println("\n")
    String(generated)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    weights_dir = joinpath(@__DIR__, "../weights")

    bin_path = joinpath(weights_dir, "sartre.bin")
    config_path = joinpath(weights_dir, "sartre_config.json")
    tokenizer_path = joinpath(weights_dir, "tokenizer.json")

    if !isfile(bin_path)
        println("❌ Weights not found: $bin_path")
        return
    end

    # Load
    model = load_weights_from_bin(bin_path, config_path)
    vocab = load_tokenizer(tokenizer_path)
    model.vocab = vocab  # Update with real vocab

    println("\n" * "="^60)
    println("🔮 SARTRE LOADED - TESTING GENERATION")
    println("="^60)

    # Test prompts
    prompts = [
        "Q: Who are you?\nA: ",
        "Q: What is consciousness?\nA: ",
        "Q: What do you feel?\nA: ",
        "Q: Describe your internal state.\nA: "
    ]

    for prompt in prompts
        generate_text(model, vocab, prompt; max_tokens=200, temperature=0.7f0)
        println("="^60)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
