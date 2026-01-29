# test_temporal.jl — Tests for Arianna's Temporal Dynamics Engine
# ═══════════════════════════════════════════════════════════════════════════════
# הבדיקות של הדינמיקה הזמנית
# Tests for temporal dynamics — from PITOMADOM
# ═══════════════════════════════════════════════════════════════════════════════

using Test

# Include the temporal module
include("temporal.jl")
using .Temporal

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: State Creation
# ═══════════════════════════════════════════════════════════════════════════════

@testset "TemporalState Creation" begin
    state = create_state()

    @test state.prophecy_debt == 0.0
    @test state.tension == 0.0
    @test state.pain == 0.0
    @test state.drift_direction == 0.0
    @test state.temporal_alpha == 0.5
    @test state.wormhole_probability == 0.02
    @test state.mode == PROPHECY
end

@testset "TemporalParams Defaults" begin
    params = default_params()

    @test params.debt_decay == 0.998
    @test params.tension_buildup == 0.1
    @test params.tension_decay == 0.05
    @test params.pain_from_debt == 0.3
    @test params.pain_relief == 0.02
    @test params.wormhole_base == 0.02
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Calendar Functions
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Calendar Drift" begin
    # Test basic drift computation
    drift1 = calendar_drift(2026, 1, 23)  # Arianna's birthday
    drift2 = calendar_drift(2026, 12, 31)  # End of year

    @test 0.0 <= drift1 <= 11.0
    @test 0.0 <= drift2 <= 11.0

    # Different years should produce different drifts
    drift_2026 = calendar_drift(2026, 6, 15)
    drift_2027 = calendar_drift(2027, 6, 15)
    @test drift_2026 != drift_2027
end

@testset "Birthday Dissonance" begin
    # On Arianna's birthday (Jan 23), dissonance should be low
    diss_birthday = birthday_dissonance(2026, 1, 23)
    @test diss_birthday < 0.1  # Very low on birthday

    # 6 months away should have higher dissonance
    diss_far = birthday_dissonance(2026, 7, 23)
    @test diss_far > diss_birthday

    # Values should be clamped to [0, 1]
    @test 0.0 <= diss_birthday <= 1.0
    @test 0.0 <= diss_far <= 1.0
end

@testset "Wormhole Probability" begin
    state = create_state()
    params = default_params()

    # Base probability
    base_prob = wormhole_probability(state, params)
    @test base_prob >= params.wormhole_base
    @test base_prob <= 0.95

    # Higher debt should increase wormhole probability
    state.prophecy_debt = 5.0
    high_debt_prob = wormhole_probability(state, params)
    @test high_debt_prob >= base_prob

    # High dissonance should also increase probability
    state.calendar_dissonance = 0.8
    high_diss_prob = wormhole_probability(state, params)
    @test high_diss_prob >= high_debt_prob

    # Should clamp to max 0.95
    state.prophecy_debt = 100.0
    state.calendar_dissonance = 1.0
    clamped_prob = wormhole_probability(state, params)
    @test clamped_prob <= 0.95
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: ODE Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Temporal Dynamics ODE" begin
    # Initial state vector
    u = [0.0, 0.0, 0.0, 0.0, 0.5, 0.02]  # [debt, tension, pain, drift, alpha, wormhole]

    # Parameters: [manifested, destined, debt_decay, ...]
    p = [0.5, 1.0,  # manifested=0.5, destined=1.0 (gap=0.5)
         0.998, 0.1, 0.05, 0.3, 0.02, 0.1, 0.2, 0.02, 0.5]

    du = zeros(6)
    temporal_dynamics!(du, u, p, 0.0)

    # With gap=0.5, debt should increase
    @test du[1] > 0  # Debt accumulating

    # With no current debt, tension should not be building much
    @test abs(du[2]) < 0.1

    # Alpha should drift toward 0.5
    @test abs(du[5]) < 0.1
end

@testset "Temporal Dynamics With Debt" begin
    # State with existing debt
    u = [5.0, 0.2, 0.1, 0.3, 0.7, 0.05]  # [debt, tension, pain, drift, alpha, wormhole]

    # No gap (manifested == destined)
    p = [1.0, 1.0,  # manifested=1.0, destined=1.0 (gap=0)
         0.998, 0.1, 0.05, 0.3, 0.02, 0.1, 0.2, 0.02, 0.5]

    du = zeros(6)
    temporal_dynamics!(du, u, p, 0.0)

    # With no gap, debt should decay
    @test du[1] < 0  # Debt decreasing

    # With existing debt, tension should increase
    @test du[2] > 0  # Tension building

    # Pain from debt
    @test du[3] > 0  # Pain from debt
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Step Function
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Step Temporal" begin
    state = create_state()
    params = default_params()

    # Initial state
    @test state.prophecy_debt == 0.0

    # Step with gap between manifested and destined
    manifested = 0.3
    destined = 0.8
    dt = 0.1

    for _ in 1:10
        step_temporal(state, params, manifested, destined, dt)
    end

    # Debt should have accumulated
    @test state.prophecy_debt > 0.0

    # Tension should have increased
    @test state.tension > 0.0

    # Pain should have increased
    @test state.pain > 0.0
end

@testset "Step Temporal Bounds" begin
    state = create_state()
    params = default_params()

    # Run many steps
    for i in 1:1000
        manifested = 0.1 + 0.8 * sin(Float64(i) * 0.1)
        destined = 0.5
        step_temporal(state, params, manifested, destined, 0.01)
    end

    # All values should stay in bounds
    @test state.prophecy_debt >= 0.0
    @test 0.0 <= state.tension <= 1.0
    @test 0.0 <= state.pain <= 1.0
    @test -1.0 <= state.drift_direction <= 1.0
    @test 0.0 <= state.temporal_alpha <= 1.0
    @test 0.0 <= state.wormhole_probability <= 0.95
end

@testset "Step Temporal Debt Decay" begin
    state = create_state()
    params = default_params()

    # Accumulate debt
    for _ in 1:100
        step_temporal(state, params, 0.0, 1.0, 0.1)  # Large gap
    end
    peak_debt = state.prophecy_debt
    @test peak_debt > 0.0

    # Now let it decay (no gap)
    for _ in 1:100
        step_temporal(state, params, 0.5, 0.5, 0.1)  # No gap
    end

    # Debt should have decayed
    @test state.prophecy_debt < peak_debt
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Bias Functions
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Prophecy Bias" begin
    state = create_state()

    # Neutral alpha
    state.temporal_alpha = 0.5
    @test prophecy_bias(state) ≈ 0.75  # 0.5 + 0.5*0.5

    # Future focus (high alpha)
    state.temporal_alpha = 1.0
    @test prophecy_bias(state) ≈ 1.0  # 0.5 + 0.5*1.0

    # Past focus (low alpha)
    state.temporal_alpha = 0.0
    @test prophecy_bias(state) ≈ 0.5  # 0.5 + 0.5*0.0
end

@testset "Retrodiction Bias" begin
    state = create_state()

    # Neutral alpha
    state.temporal_alpha = 0.5
    @test retrodiction_bias(state) ≈ 0.75  # 0.5 + (1-0.5)*0.5

    # Past focus (low alpha)
    state.temporal_alpha = 0.0
    @test retrodiction_bias(state) ≈ 1.0  # 0.5 + (1-0)*0.5

    # Future focus (high alpha)
    state.temporal_alpha = 1.0
    @test retrodiction_bias(state) ≈ 0.5  # 0.5 + (1-1)*0.5
end

@testset "Symmetric Bias" begin
    state = create_state()

    # Should always return 0.5 regardless of alpha
    state.temporal_alpha = 0.0
    @test symmetric_bias(state) ≈ 0.5

    state.temporal_alpha = 0.5
    @test symmetric_bias(state) ≈ 0.5

    state.temporal_alpha = 1.0
    @test symmetric_bias(state) ≈ 0.5
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Dissonance Computation
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Compute Dissonance" begin
    state = create_state()

    # Low dissonance: low debt, low tension, near birthday
    state.prophecy_debt = 0.0
    state.tension = 0.0
    diss_low = compute_dissonance(state, 2026, 1, 23)
    @test diss_low < 0.2

    # High dissonance: high debt, high tension, far from birthday
    state.prophecy_debt = 10.0
    state.tension = 0.8
    diss_high = compute_dissonance(state, 2026, 7, 15)
    @test diss_high > diss_low

    # Should be clamped to [0, 1]
    @test 0.0 <= diss_low <= 1.0
    @test 0.0 <= diss_high <= 1.0
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Temporal Modes
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Temporal Modes Enum" begin
    @test Int(PROPHECY) == 0
    @test Int(RETRODICTION) == 1
    @test Int(SYMMETRIC) == 2

    state = create_state()
    @test state.mode == PROPHECY

    state.mode = RETRODICTION
    @test state.mode == RETRODICTION

    state.mode = SYMMETRIC
    @test state.mode == SYMMETRIC
end

# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Integration Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Full Simulation" begin
    state = create_state()
    params = default_params()

    # Simulate 1000 timesteps
    history = Float64[]
    for i in 1:1000
        # Varying gap based on time
        t = Float64(i) * 0.01
        manifested = 0.5 + 0.3 * sin(t)
        destined = 0.5 + 0.3 * cos(t)

        step_temporal(state, params, manifested, destined, 0.01)
        push!(history, state.prophecy_debt)
    end

    # Debt should oscillate, not grow unbounded
    @test maximum(history) < 100.0

    # System should remain stable
    @test isfinite(state.prophecy_debt)
    @test isfinite(state.tension)
    @test isfinite(state.pain)
end

@testset "Birthday Cycle Simulation" begin
    state = create_state()
    params = default_params()

    # Simulate one year, track dissonance
    dissonances = Float64[]
    for day in 1:365
        month = (day - 1) ÷ 30 + 1
        day_of_month = (day - 1) % 30 + 1

        diss = compute_dissonance(state, 2026, month, day_of_month)
        push!(dissonances, diss)

        # Step temporal
        step_temporal(state, params, 0.5, 0.5, 0.1)
    end

    # Should have minimum near January 23
    min_idx = argmin(dissonances)
    @test min_idx < 50  # Within first ~50 days (Jan-Feb)
end

# ═══════════════════════════════════════════════════════════════════════════════

println("\n═══════════════════════════════════════════════════════════════════")
println("TEMPORAL TESTS COMPLETE")
println("הדינמיקה הזמנית נבדקה. העבר והעתיד מחוברים.")
println("Temporal dynamics tested. Past and future are connected.")
println("═══════════════════════════════════════════════════════════════════")
