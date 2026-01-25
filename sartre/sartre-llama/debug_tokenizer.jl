#!/usr/bin/env julia
# Debug tokenizer

push!(LOAD_PATH, @__DIR__)

include("sartre_transformer.jl")
using .SARTRETransformer: Vocabulary

using JSON3

function load_tokenizer(tokenizer_path::String)
    tok_json = JSON3.read(read(tokenizer_path, String))

    char_to_idx = Dict{Char, Int}()
    idx_to_char = Dict{Int, Char}()

    println("Loading tokenizer...")
    for (key, idx) in tok_json.char_to_id
        char_str = String(key)
        char = first(char_str)
        # Python is 0-indexed, Julia is 1-indexed
        char_to_idx[char] = idx + 1
        idx_to_char[idx + 1] = char
        if length(char_to_idx) <= 10
            println("  '$char' -> $(idx+1)")
        end
    end

    vocab_size = tok_json.vocab_size
    println("Vocab size: $vocab_size")

    Vocabulary(char_to_idx, idx_to_char, vocab_size)
end

vocab = load_tokenizer("../weights/tokenizer.json")

# Test encode/decode
test_str = "Q: Who are you?\nA: I see"

println("\nTest string: \"$test_str\"")

# Encode
tokens = Int[]
for c in test_str
    if haskey(vocab.char_to_idx, c)
        push!(tokens, vocab.char_to_idx[c])
    else
        println("WARNING: char '$c' not in vocab!")
    end
end

println("Tokens: $tokens")

# Decode
decoded_chars = Char[]
for t in tokens
    if haskey(vocab.idx_to_char, t)
        push!(decoded_chars, vocab.idx_to_char[t])
    end
end
decoded = String(decoded_chars)

println("Decoded: \"$decoded\"")
println("Match: $(test_str == decoded)")
