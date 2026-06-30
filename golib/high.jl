const EMOTIONAL_WEIGHTS = Dict{String,Float64}(
	"great" => 0.8, "love" => 0.9, "amazing" => 0.7, "wonderful" => 0.8, "excellent" => 0.7,
	"beautiful" => 0.8, "fantastic" => 0.7, "awesome" => 0.8, "perfect" => 0.7, "brilliant" => 0.8,
	"happy" => 0.7, "joy" => 0.8, "excited" => 0.7, "delighted" => 0.8, "pleased" => 0.6,
	"good" => 0.5, "nice" => 0.4, "fine" => 0.3, "okay" => 0.1, "thanks" => 0.4,
	"grateful" => 0.7, "blessed" => 0.6, "peaceful" => 0.5, "calm" => 0.4, "serene" => 0.5,
	"hope" => 0.6, "dream" => 0.5, "inspire" => 0.6, "create" => 0.5, "grow" => 0.4,
	"terrible" => -0.8, "hate" => -0.9, "awful" => -0.7, "horrible" => -0.8, "disgusting" => -0.9,
	"sad" => -0.6, "angry" => -0.7, "frustrated" => -0.6, "disappointed" => -0.6, "upset" => -0.6,
	"bad" => -0.5, "wrong" => -0.4, "fail" => -0.6, "lose" => -0.5, "hurt" => -0.7,
	"pain" => -0.8, "suffer" => -0.8, "fear" => -0.7, "anxiety" => -0.6, "stress" => -0.5,
	"alone" => -0.6, "lonely" => -0.7, "empty" => -0.5, "nothing" => -0.6, "worthless" => -0.9,
	"stupid" => -0.7, "ugly" => -0.6, "weak" => -0.5, "useless" => -0.8, "pathetic" => -0.8,
	"отлично" => 0.8, "классно" => 0.7, "супер" => 0.8, "круто" => 0.7, "прекрасно" => 0.8,
	"здорово" => 0.7, "замечательно" => 0.8, "чудесно" => 0.7, "великолепно" => 0.8,
	"люблю" => 0.9, "радость" => 0.8, "счастье" => 0.9, "мир" => 0.5, "добро" => 0.6,
	"красиво" => 0.7, "хорошо" => 0.5, "спасибо" => 0.5, "благодарю" => 0.6,
	"ужасно" => -0.8, "плохо" => -0.6, "грустно" => -0.6, "злой" => -0.7, "расстроен" => -0.6,
	"больно" => -0.8, "страшно" => -0.7, "одиноко" => -0.7, "пусто" => -0.5, "ничто" => -0.6,
	"ненавижу" => -0.9, "страдаю" => -0.8, "боюсь" => -0.7, "тревога" => -0.6,
	"глупый" => -0.6, "слабый" => -0.5, "никчёмный" => -0.8, "жалкий" => -0.7,
	"טוב" => 0.5, "יפה" => 0.7, "מדהים" => 0.8, "נהדר" => 0.7, "אהבה" => 0.9,
	"שמחה" => 0.8, "תקווה" => 0.6, "שלום" => 0.5, "ברכה" => 0.6,
	"רע" => -0.5, "נורא" => -0.8, "עצוב" => -0.6, "כועס" => -0.7, "פחד" => -0.7,
	"כאב" => -0.8, "בודד" => -0.7, "ריק" => -0.5, "שנאה" => -0.9,
	"die" => -0.9, "kill" => -0.9, "failure" => -0.8, "loser" => -0.8,
	"reject" => -0.7, "abandon" => -0.8, "betray" => -0.8, "forget" => -0.5,
	"ignore" => -0.6, "invisible" => -0.7, "broken" => -0.7, "damaged" => -0.7,
	"ruined" => -0.7, "trapped" => -0.7, "hopeless" => -0.8, "lost" => -0.5,
)
# Arianna High Mathematical Brain — faithful Julia port of legacy inner_world/high.go HighMathEngine.
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
_emo(s) = begin
    t = 0.0; c = 0
    for w in tokenize(s); if haskey(EMOTIONAL_WEIGHTS,w); t += EMOTIONAL_WEIGHTS[w]; c += 1; end; end
    (t, c)
end
analyze_valence(s::AbstractString) = (p=_emo(s); p[2]==0 ? 0.0 : p[1]/p[2])
analyze_arousal(s::AbstractString) = (p=_emo(s); v = p[2]==0 ? 0.0 : p[1]/p[2]; clamp(abs(v)+p[2]*0.01, 0.0, 1.0))

# ---- VectorizedEntropy: word-level Shannon entropy modulated by emotional intensity (legacy VectorizedEntropy) ----
function _vec_entropy(s::AbstractString)
    words = tokenize(s)
    isempty(words) && return (0.0, 0.0)
    wc = Dict{String,Int}(); total = 0; esum = 0.0
    for w in words
        wc[w] = get(wc, w, 0) + 1; total += 1
        haskey(EMOTIONAL_WEIGHTS, w) && (esum += EMOTIONAL_WEIGHTS[w])
    end
    total == 0 && return (0.0, 0.0)
    ent = 0.0
    for (_, c) in wc; p = c / total; p > 0 && (ent -= p * log2(p)); end
    escore = esum / total
    ent *= (1.0 + abs(escore) * 0.2)
    (ent, escore)
end
vectorized_entropy(s::AbstractString) = _vec_entropy(s)[1]
emotional_score(s::AbstractString) = _vec_entropy(s)[2]

# ---- EmotionalAlignment (legacy EmotionalAlignment) ----
function emotional_alignment(a::AbstractString, b::AbstractString)
    e1 = emotional_score(a); e2 = emotional_score(b)
    if (e1 >= 0) == (e2 >= 0)
        diff = abs(e1 - e2); mx = max(abs(e1), abs(e2))
        mx == 0 && return 1.0
        return 1.0 - diff / mx
    end
    -abs(e1 - e2) / 2
end

# ---- PredictiveSurprise: free-energy proxy (legacy PredictiveSurprise) ----
function predictive_surprise(expected::AbstractString, actual::AbstractString)
    se = semantic_distance(expected, actual)
    ee = abs(1.0 - emotional_alignment(expected, actual))
    en = abs(vectorized_entropy(expected) - vectorized_entropy(actual))
    se * 0.4 + ee * 0.4 + en * 0.2
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

# ---- TextRhythm (legacy TextRhythm): syllable avg / variance / pause density ----
const _VOWELS = Set(collect("aeiouyаеёиоуыэюяאֵֶֻוִ"))
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
        c = 0.0
        for ch in w; ch in _VOWELS && (c += 1); end
        c < 1 && (c = 1.0)
        push!(syl, c)
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
