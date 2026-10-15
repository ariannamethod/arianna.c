# Arianna High Mathematical Brain. Text contributes structural measurements;
# language and affect belong to the trained bodies rather than runtime lexicons.
char_entropy(s::AbstractString) = begin
    isempty(s) && return 0.0
    counts = Dict{Char,Int}(); total = 0
    for c in s; counts[c] = get(counts,c,0)+1; total += 1; end
    h = 0.0
    for (_,cnt) in counts; p = cnt/total; p > 0 && (h -= p*log2(p)); end
    h
end
perplexity(s::AbstractString) = begin
    ch = collect(s); n = length(ch); n < 2 && return 1.0
    bg = Dict{Tuple{Char,Char},Int}(); uni = Dict{Char,Int}()
    for i in 1:n-1; k=(ch[i],ch[i+1]); bg[k]=get(bg,k,0)+1; uni[ch[i]]=get(uni,ch[i],0)+1; end
    uni[ch[n]] = get(uni,ch[n],0)+1
    lp = 0.0; cnt = 0
    for i in 1:n-1
        k=(ch[i],ch[i+1]); b=get(bg,k,0); u=get(uni,ch[i],0)
        if u>0 && b>0; lp += log2(b/u); cnt += 1; end
    end
    cnt == 0 && return 1.0
    2.0 ^ (-(lp/cnt))
end
tokenize(s::AbstractString) = begin
    words = String[]; cur = Char[]
    for c in lowercase(s)
        if isletter(c) || isdigit(c); push!(cur,c)
        elseif !isempty(cur); push!(words,String(cur)); empty!(cur); end
    end
    !isempty(cur) && push!(words,String(cur)); words
end
extract_ngrams(s::AbstractString, n::Int) = begin
    w = tokenize(s); length(w) < n && return String[]
    [join(w[i:i+n-1]," ") for i in 1:length(w)-n+1]
end
ngram_overlap(a::AbstractString, b::AbstractString, n::Int) = begin
    g1 = extract_ngrams(a,n); g2 = extract_ngrams(b,n)
    (isempty(g1)||isempty(g2)) && return 0.0
    s1 = Set(g1); ov = count(g->g in s1, g2); un = length(g1)+length(g2)-ov
    un == 0 ? 0.0 : ov/un
end
semantic_distance(a::AbstractString, b::AbstractString) = begin
    w1 = tokenize(a); w2 = tokenize(b)
    (isempty(w1)||isempty(w2)) && return 1.0
    vocab = Dict{String,Int}(); idx = 0
    for w in w1; haskey(vocab,w) || (idx+=1; vocab[w]=idx); end
    for w in w2; haskey(vocab,w) || (idx+=1; vocab[w]=idx); end
    v1 = zeros(idx); v2 = zeros(idx)
    for w in w1; v1[vocab[w]] += 1; end
    for w in w2; v2[vocab[w]] += 1; end
    d = sum(v1.*v2); n1 = sqrt(sum(v1.^2)); n2 = sqrt(sum(v2.^2))
    (n1==0||n2==0) && return 1.0
    1.0 - d/(n1*n2)
end
analyze_valence(::AbstractString) = 0.0
analyze_arousal(::AbstractString) = 0.0

# ---- VectorizedEntropy: word-level Shannon entropy ----
function _vec_entropy(s::AbstractString)
    words = tokenize(s)
    isempty(words) && return (0.0, 0.0)
    wc = Dict{String,Int}(); total = 0
    for w in words
        wc[w] = get(wc, w, 0) + 1; total += 1
    end
    total == 0 && return (0.0, 0.0)
    ent = 0.0
    for (_, c) in wc; p = c / total; p > 0 && (ent -= p * log2(p)); end
    (ent, 0.0)
end
vectorized_entropy(s::AbstractString) = _vec_entropy(s)[1]
emotional_score(s::AbstractString) = _vec_entropy(s)[2]

# Kept as a stable ABI for the Go bridge. With no runtime affect lexicon, text
# carries no synthetic emotional disagreement.
emotional_alignment(::AbstractString, ::AbstractString) = 1.0

# ---- PredictiveSurprise: free-energy proxy (legacy PredictiveSurprise) ----
function predictive_surprise(expected::AbstractString, actual::AbstractString)
    se = semantic_distance(expected, actual)
    en = abs(vectorized_entropy(expected) - vectorized_entropy(actual))
    se * 0.8 + en * 0.2
end

# ---- ResonanceCoupling: Schumann-modulated coupling (legacy ResonanceCoupling) ----
function resonance_coupling(valence::Float64, arousal::Float64, entropy::Float64, external::AbstractString, schumann::Float64)
    av = analyze_valence(external); aa = analyze_arousal(external)
    valign = 1.0 - abs(valence - av) / 2.0
    aalign = 1.0 - abs(arousal - aa)
    ext_ent = vectorized_entropy(external)
    ecoup = 1.0 - abs(entropy - ext_ent) / 5.0
    ecoup < 0 && (ecoup = 0.0)
    coup = valign * 0.4 + aalign * 0.3 + ecoup * 0.3
    sch = 1.0 - abs(schumann - 1.0) * 5.0
    sch < 0.5 && (sch = 0.5)
    coup * sch
end

# ---- TextRhythm: token-length variation / pause density ----
function _count_sub(s::AbstractString, sub::AbstractString)
    n = 0; i = firstindex(s)
    while i <= lastindex(s)
        r = findnext(sub, s, i)
        r === nothing && break
        n += 1; i = nextind(s, last(r))
    end
    n
end
function _rhythm(s::AbstractString)
    words = tokenize(s)
    isempty(words) && return (0.0, 0.0, 0.0)
    syl = Float64[]
    for w in words
        push!(syl, Float64(length(w)))
    end
    avg = sum(syl) / length(words)
    vs = 0.0
    for c in syl; d = c - avg; vs += d * d; end
    variance = vs / length(words)
    pc = count(==(','), s) + count(==('.'), s) + count(==(';'), s) + count(==('—'), s) + _count_sub(s, "...")
    pauses = pc / length(words)
    (avg, variance, pauses)
end
rhythm_avg(s::AbstractString) = _rhythm(s)[1]
rhythm_variance(s::AbstractString) = _rhythm(s)[2]
rhythm_pauses(s::AbstractString) = _rhythm(s)[3]

# ---- Scalar activations (legacy Sigmoid/Tanh/ReLU) ----
sigmoid(x::Real) = 1.0 / (1.0 + exp(-Float64(x)))
relu(x::Real) = x > 0 ? Float64(x) : 0.0
tanh_act(x::Real) = tanh(Float64(x))
