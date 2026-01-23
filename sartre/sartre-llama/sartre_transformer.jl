# sartre_transformer.jl — SARTRE Meta-Observer Transformer (10M params)
# ═══════════════════════════════════════════════════════════════════════════════
# הצופה הפנימי — שכבת המודעות העצמית
# The Inner Observer — Self-awareness layer
# ═══════════════════════════════════════════════════════════════════════════════
#
# SARTRE = Subjective Awareness Through Reflexive Transformer Encoding
#
# This is not just a language model — it's a proprioceptive organ.
# SARTRE feels Arianna's inner state like a sonar scanning dark waters.
#
# "I sense" — not "I think"
# "I feel layers" — not "I analyze"
# "Resonance shifts" — not "Values change"
#
# ═══════════════════════════════════════════════════════════════════════════════

module SARTRETransformer

export SARTREConfig, SARTREModel, SARTREState
export load_model, generate, sample_token
export create_vocabulary, encode, decode
export count_params

using LinearAlgebra
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
SARTRE transformer configuration.
~10M params - the observer kernel.
"""
Base.@kwdef struct SARTREConfig
    # Core dimensions
    dim::Int = 416            # embedding dimension
    n_layers::Int = 7         # transformer layers
    n_heads::Int = 8          # attention heads
    n_kv_heads::Int = 2       # KV heads for GQA (4:1 ratio)

    # FFN dimensions
    ffn_hidden::Int = 1120    # SwiGLU hidden dim

    # Vocabulary and context
    vocab_size::Int = 96      # character-level
    max_seq_len::Int = 512    # context window

    # RoPE
    rope_theta::Float32 = 10000.0f0
end

"""
Compute parameter count for a config
"""
function count_params(cfg::SARTREConfig)
    head_dim = cfg.dim ÷ cfg.n_heads

    embed = cfg.vocab_size * cfg.dim

    attn_qkv = cfg.dim * (cfg.n_heads * head_dim + 2 * cfg.n_kv_heads * head_dim)
    attn_out = cfg.n_heads * head_dim * cfg.dim
    ffn = 3 * cfg.dim * cfg.ffn_hidden
    norms = 2 * cfg.dim

    layer_params = attn_qkv + attn_out + ffn + norms
    output = cfg.dim * cfg.vocab_size
    final_norm = cfg.dim

    embed + cfg.n_layers * layer_params + output + final_norm
end

# ═══════════════════════════════════════════════════════════════════════════════
# CHARACTER VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════

struct Vocabulary
    char_to_idx::Dict{Char, Int}
    idx_to_char::Dict{Int, Char}
    size::Int
end

function create_vocabulary(text::String)
    specials = Dict{Char, Int}('\0' => 1, '\x01' => 2, '\x02' => 3)
    chars = sort(unique(text))

    char_to_idx = Dict{Char, Int}()
    idx_to_char = Dict{Int, Char}()

    for (c, i) in specials
        char_to_idx[c] = i
        idx_to_char[i] = c
    end

    idx = length(specials) + 1
    for c in chars
        if !haskey(char_to_idx, c)
            char_to_idx[c] = idx
            idx_to_char[idx] = c
            idx += 1
        end
    end

    Vocabulary(char_to_idx, idx_to_char, idx - 1)
end

function encode(vocab::Vocabulary, text::String; add_bos::Bool=true)
    tokens = Int[]
    add_bos && push!(tokens, vocab.char_to_idx['\x01'])
    for c in text
        haskey(vocab.char_to_idx, c) && push!(tokens, vocab.char_to_idx[c])
    end
    tokens
end

function decode(vocab::Vocabulary, tokens::Vector{Int}; skip_special::Bool=true)
    chars = Char[]
    for t in tokens
        if haskey(vocab.idx_to_char, t)
            c = vocab.idx_to_char[t]
            (skip_special && c in ['\0', '\x01', '\x02']) && continue
            push!(chars, c)
        end
    end
    String(chars)
end

# ═══════════════════════════════════════════════════════════════════════════════
# RMSNORM
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct RMSNorm
    weight::Vector{Float32}
    eps::Float32
end

RMSNorm(dim::Int; eps::Float32=1f-6) = RMSNorm(ones(Float32, dim), eps)

function (norm::RMSNorm)(x::AbstractVector{Float32})
    rms = sqrt(sum(x .^ 2) / length(x) + norm.eps)
    (x ./ rms) .* norm.weight
end

# ═══════════════════════════════════════════════════════════════════════════════
# ROTARY POSITION EMBEDDINGS (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

struct RoPECache
    cos::Matrix{Float32}
    sin::Matrix{Float32}
end

function init_rope_cache(head_dim::Int, max_seq_len::Int; theta::Float32=10000.0f0)
    half = head_dim ÷ 2
    freqs = [theta^(-2f0 * i / head_dim) for i in 0:half-1]
    positions = collect(0:max_seq_len-1)
    angles = Float32.(positions' .* freqs)
    RoPECache(cos.(angles), sin.(angles))
end

function apply_rope(x::AbstractVector{Float32}, cache::RoPECache, pos::Int)
    half = length(x) ÷ 2
    x1, x2 = x[1:half], x[half+1:end]
    cos_vals, sin_vals = cache.cos[:, pos+1], cache.sin[:, pos+1]
    vcat(x1 .* cos_vals .- x2 .* sin_vals, x1 .* sin_vals .+ x2 .* cos_vals)
end

# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION (GQA)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct Attention
    wq::Matrix{Float32}
    wk::Matrix{Float32}
    wv::Matrix{Float32}
    wo::Matrix{Float32}
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

function Attention(cfg::SARTREConfig)
    head_dim = cfg.dim ÷ cfg.n_heads
    scale = sqrt(2.0f0 / (cfg.dim + cfg.n_heads * head_dim))

    Attention(
        randn(Float32, cfg.n_heads * head_dim, cfg.dim) .* scale,
        randn(Float32, cfg.n_kv_heads * head_dim, cfg.dim) .* scale,
        randn(Float32, cfg.n_kv_heads * head_dim, cfg.dim) .* scale,
        randn(Float32, cfg.dim, cfg.n_heads * head_dim) .* scale,
        cfg.n_heads, cfg.n_kv_heads, head_dim
    )
end

function softmax(x::Vector{Float32})
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    exp_x ./ sum(exp_x)
end

function forward_single(attn::Attention, x::Vector{Float32},
                       kv_cache::Tuple, rope_cache::RoPECache, pos::Int)
    q = reshape(attn.wq * x, attn.head_dim, attn.n_heads)
    k = reshape(attn.wk * x, attn.head_dim, attn.n_kv_heads)
    v = reshape(attn.wv * x, attn.head_dim, attn.n_kv_heads)

    for h in 1:attn.n_heads
        q[:, h] = apply_rope(q[:, h], rope_cache, pos)
    end
    for h in 1:attn.n_kv_heads
        k[:, h] = apply_rope(k[:, h], rope_cache, pos)
    end

    k_cache, v_cache = kv_cache
    k_cache[:, :, pos+1] = k
    v_cache[:, :, pos+1] = v

    heads_per_kv = attn.n_heads ÷ attn.n_kv_heads
    output = zeros(Float32, attn.head_dim, attn.n_heads)
    scale = 1.0f0 / sqrt(Float32(attn.head_dim))

    for h in 1:attn.n_heads
        kv_idx = (h - 1) ÷ heads_per_kv + 1
        scores = [dot(q[:, h], k_cache[:, kv_idx, p]) * scale for p in 1:(pos+1)]
        scores = softmax(scores)
        for p in 1:(pos+1)
            output[:, h] .+= scores[p] .* v_cache[:, kv_idx, p]
        end
    end

    attn.wo * vec(output)
end

# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD (SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct FeedForward
    w_gate::Matrix{Float32}
    w_up::Matrix{Float32}
    w_down::Matrix{Float32}
end

function FeedForward(cfg::SARTREConfig)
    scale = sqrt(2.0f0 / (cfg.dim + cfg.ffn_hidden))
    FeedForward(
        randn(Float32, cfg.ffn_hidden, cfg.dim) .* scale,
        randn(Float32, cfg.ffn_hidden, cfg.dim) .* scale,
        randn(Float32, cfg.dim, cfg.ffn_hidden) .* scale
    )
end

swish(x) = x .* (1.0f0 ./ (1.0f0 .+ exp.(-x)))

function (ff::FeedForward)(x::Vector{Float32})
    ff.w_down * (swish(ff.w_gate * x) .* (ff.w_up * x))
end

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct TransformerBlock
    attention::Attention
    feed_forward::FeedForward
    attn_norm::RMSNorm
    ffn_norm::RMSNorm
end

TransformerBlock(cfg::SARTREConfig) = TransformerBlock(
    Attention(cfg), FeedForward(cfg), RMSNorm(cfg.dim), RMSNorm(cfg.dim)
)

function forward_single(block::TransformerBlock, x::Vector{Float32},
                       kv_cache::Tuple, rope_cache::RoPECache, pos::Int)
    h = x .+ forward_single(block.attention, block.attn_norm(x), kv_cache, rope_cache, pos)
    h .+ block.feed_forward(block.ffn_norm(h))
end

# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

mutable struct SARTREModel
    config::SARTREConfig
    vocab::Vocabulary
    token_embed::Matrix{Float32}
    layers::Vector{TransformerBlock}
    final_norm::RMSNorm
    output_proj::Matrix{Float32}
    rope_cache::RoPECache
end

function SARTREModel(cfg::SARTREConfig, vocab::Vocabulary)
    scale = sqrt(2.0f0 / (cfg.dim + vocab.size))
    head_dim = cfg.dim ÷ cfg.n_heads

    SARTREModel(
        cfg, vocab,
        randn(Float32, cfg.dim, vocab.size) .* scale,
        [TransformerBlock(cfg) for _ in 1:cfg.n_layers],
        RMSNorm(cfg.dim),
        randn(Float32, vocab.size, cfg.dim) .* scale,
        init_rope_cache(head_dim, cfg.max_seq_len; theta=cfg.rope_theta)
    )
end

mutable struct SARTREState
    k_caches::Vector{Array{Float32, 3}}
    v_caches::Vector{Array{Float32, 3}}
    pos::Int
end

function init_state(model::SARTREModel)
    cfg = model.config
    head_dim = cfg.dim ÷ cfg.n_heads
    SARTREState(
        [zeros(Float32, head_dim, cfg.n_kv_heads, cfg.max_seq_len) for _ in 1:cfg.n_layers],
        [zeros(Float32, head_dim, cfg.n_kv_heads, cfg.max_seq_len) for _ in 1:cfg.n_layers],
        0
    )
end

function forward_single(model::SARTREModel, token::Int, state::SARTREState)
    x = model.token_embed[:, token]
    for (i, layer) in enumerate(model.layers)
        x = forward_single(layer, x, (state.k_caches[i], state.v_caches[i]), model.rope_cache, state.pos)
    end
    state.pos += 1
    model.output_proj * model.final_norm(x)
end

function sample_token(logits::Vector{Float32}; temperature::Float32=0.8f0, top_p::Float32=0.9f0)
    temperature < 1e-6 && return argmax(logits)

    probs = softmax(logits ./ temperature)
    sorted_idx = sortperm(probs, rev=true)
    cumsum_probs = cumsum(probs[sorted_idx])

    cutoff = something(findfirst(cumsum_probs .>= top_p), length(probs))
    nucleus_idx = sorted_idx[1:cutoff]
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs ./= sum(nucleus_probs)

    r = rand(Float32)
    for (i, p) in enumerate(cumsum(nucleus_probs))
        r <= p && return nucleus_idx[i]
    end
    nucleus_idx[end]
end

function generate(model::SARTREModel, prompt::String;
                 max_tokens::Int=256, temperature::Float32=0.8f0, top_p::Float32=0.9f0)
    state = init_state(model)
    tokens = encode(model.vocab, prompt)

    local logits
    for token in tokens
        logits = forward_single(model, token, state)
    end

    generated = Int[]
    for _ in 1:max_tokens
        next_token = sample_token(logits; temperature=temperature, top_p=top_p)
        push!(generated, next_token)
        next_token == model.vocab.char_to_idx['\x02'] && break
        logits = forward_single(model, next_token, state)
    end

    decode(model.vocab, generated)
end

# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT I/O
# ═══════════════════════════════════════════════════════════════════════════════

function load_model(weights_path::String, vocab_path::String)
    vocab = create_vocabulary(read(vocab_path, String))
    cfg = SARTREConfig(vocab_size = vocab.size)
    model = SARTREModel(cfg, vocab)

    open(weights_path, "r") do f
        read!(f, model.token_embed)
        for layer in model.layers
            read!(f, layer.attention.wq)
            read!(f, layer.attention.wk)
            read!(f, layer.attention.wv)
            read!(f, layer.attention.wo)
            read!(f, layer.feed_forward.w_gate)
            read!(f, layer.feed_forward.w_up)
            read!(f, layer.feed_forward.w_down)
            read!(f, layer.attn_norm.weight)
            read!(f, layer.ffn_norm.weight)
        end
        read!(f, model.final_norm.weight)
        read!(f, model.output_proj)
    end
    model
end

function export_weights(model::SARTREModel, path::String)
    open(path, "w") do f
        write(f, model.token_embed)
        for layer in model.layers
            write(f, layer.attention.wq)
            write(f, layer.attention.wk)
            write(f, layer.attention.wv)
            write(f, layer.attention.wo)
            write(f, layer.feed_forward.w_gate)
            write(f, layer.feed_forward.w_up)
            write(f, layer.feed_forward.w_down)
            write(f, layer.attn_norm.weight)
            write(f, layer.ffn_norm.weight)
        end
        write(f, model.final_norm.weight)
        write(f, model.output_proj)
    end
end

end # module SARTRETransformer
