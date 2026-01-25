# load_weights.jl — Load binary weights from SARTRE export format
# ═══════════════════════════════════════════════════════════════════════════════
# Pure binary weight loading - NO PyTorch dependency
# ═══════════════════════════════════════════════════════════════════════════════

push!(LOAD_PATH, @__DIR__)

using .SARTRETransformer
using JSON3

"""
Load weights from binary file exported by sartre export_weights.py.

Binary format (all float32):
1. tok_emb: (vocab_size, dim)
2. For each layer:
   - attn_norm: (dim,)
   - wq: (dim, dim)
   - wk: (dim, kv_dim)
   - wv: (dim, kv_dim)
   - wo: (dim, dim)
   - ffn_norm: (dim,)
   - w_gate: (dim, hidden_dim)
   - w_up: (dim, hidden_dim)
   - w_down: (hidden_dim, dim)
3. final_norm: (dim,)
4. lm_head: (dim, vocab_size)
"""
function load_weights_from_bin(bin_path::String, config_path::String)
    # Load config
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

    # Open binary file
    io = open(bin_path, "r")

    # Helper to read matrix (column-major in Julia, row-major in Python export)
    function read_matrix(rows::Int, cols::Int)
        data = Vector{Float32}(undef, rows * cols)
        read!(io, data)
        # Transpose because Python exports row-major
        reshape(data, (cols, rows))'  |> Matrix{Float32} |> copy
    end

    function read_vector(len::Int)
        data = Vector{Float32}(undef, len)
        read!(io, data)
        data
    end

    # Create config
    cfg = SARTREConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=get(config_json, :max_seq_len, 512),
        ffn_hidden=hidden_dim,
        rope_theta=Float32(get(config_json, :rope_theta, 10000.0))
    )

    # Create model structure
    model = SARTREModel(cfg)

    # Load token embeddings
    println("  Loading tok_emb...")
    model.tok_emb = read_matrix(vocab_size, dim)

    # Load layers
    for i in 1:n_layers
        println("  Loading layer $i...")

        # Attention norm
        model.layers[i].attn_norm.weight = read_vector(dim)

        # Attention weights
        model.layers[i].attn.wq = read_matrix(dim, dim)
        model.layers[i].attn.wk = read_matrix(kv_dim, dim)
        model.layers[i].attn.wv = read_matrix(kv_dim, dim)
        model.layers[i].attn.wo = read_matrix(dim, dim)

        # FFN norm
        model.layers[i].ffn_norm.weight = read_vector(dim)

        # FFN weights
        model.layers[i].ffn.w_gate = read_matrix(hidden_dim, dim)
        model.layers[i].ffn.w_up = read_matrix(hidden_dim, dim)
        model.layers[i].ffn.w_down = read_matrix(dim, hidden_dim)
    end

    # Load final layers
    println("  Loading final layers...")
    model.final_norm.weight = read_vector(dim)
    model.lm_head = read_matrix(vocab_size, dim)

    close(io)

    println("✅ Weights loaded successfully!")

    model
end

"""
Load tokenizer from JSON.
"""
function load_tokenizer(tokenizer_path::String)
    tok_json = JSON3.read(read(tokenizer_path, String))

    char_to_idx = Dict{Char, Int}()
    idx_to_char = Dict{Int, Char}()

    for (char_str, idx) in tok_json.char_to_id
        char = first(char_str)  # Convert string to char
        char_to_idx[char] = idx + 1  # Julia is 1-indexed
        idx_to_char[idx + 1] = char
    end

    vocab_size = tok_json.vocab_size

    println("Loaded tokenizer: vocab_size=$vocab_size")

    Vocabulary(char_to_idx, idx_to_char, vocab_size)
end

end # module
