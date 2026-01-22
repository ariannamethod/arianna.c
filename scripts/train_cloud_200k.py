#!/usr/bin/env python3
"""
Train Cloud 200K with actual backprop.

Architecture: 100→128→64→32→1 per chamber (4 layers)
- 6 chambers × ~23K = ~138K params
- 1 observer × ~41K params
- Total: ~180K ≈ 200K params

No PyTorch needed. Pure numpy.
"""

import json
import numpy as np
from pathlib import Path
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# ANCHORS
# ═══════════════════════════════════════════════════════════════════════════════

CHAMBER_NAMES = ["FEAR", "LOVE", "RAGE", "VOID", "FLOW", "COMPLEX"]

ANCHORS = {
    "FEAR": ["fear", "terror", "panic", "anxious", "dread", "horror", "unease",
             "paranoia", "worry", "nervous", "scared", "frighten", "alarm",
             "tense", "apprehension", "threat", "vulnerable", "insecure", "timid", "wary"],
    "LOVE": ["love", "warmth", "tenderness", "devotion", "longing", "yearning",
             "affection", "care", "intimacy", "attachment", "adoration", "passion",
             "fondness", "cherish", "desire", "compassion", "gentle", "sweet"],
    "RAGE": ["anger", "rage", "fury", "hatred", "spite", "disgust", "irritation",
             "frustration", "resentment", "hostility", "aggression", "bitterness",
             "contempt", "loathing", "annoyance", "outrage", "wrath"],
    "VOID": ["emptiness", "numbness", "hollow", "nothing", "absence", "void",
             "dissociation", "detachment", "apathy", "indifference", "drift",
             "blank", "flat", "dead", "cold"],
    "FLOW": ["curiosity", "surprise", "wonder", "confusion", "anticipation",
             "ambivalence", "uncertainty", "restless", "searching", "transition",
             "shift", "change", "flux", "between", "liminal"],
    "COMPLEX": ["shame", "guilt", "envy", "jealousy", "pride", "disappointment",
                "betrayal", "relief", "nostalgia", "bittersweet", "melancholy",
                "regret", "hope", "gratitude", "awe"]
}

ALL_ANCHORS = []
ANCHOR_TO_CHAMBER = {}
for chamber, words in ANCHORS.items():
    for word in words:
        ALL_ANCHORS.append(word)
        ANCHOR_TO_CHAMBER[word] = chamber

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ═══════════════════════════════════════════════════════════════════════════════
# CHAMBER MLP: 100→128→64→32→1
# ═══════════════════════════════════════════════════════════════════════════════

class ChamberMLP:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Xavier initialization
        self.W1 = np.random.randn(100, 128) * np.sqrt(2.0 / 100)
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(32)
        self.W4 = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
        self.b4 = np.zeros(1)

    def forward(self, x):
        """Forward pass with cached activations for backprop."""
        self.x = x

        self.z1 = x @ self.W1 + self.b1
        self.a1 = swish(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = swish(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = swish(self.z3)

        self.z4 = self.a3 @ self.W4 + self.b4
        self.out = sigmoid(self.z4)

        return self.out.item()

    def backward(self, grad_out, lr=0.01):
        """Backprop through all layers."""
        # Output layer
        grad_z4 = grad_out * sigmoid_derivative(self.z4)
        grad_W4 = self.a3.reshape(-1, 1) @ grad_z4.reshape(1, -1)
        grad_b4 = grad_z4
        grad_a3 = (grad_z4 @ self.W4.T).flatten()

        # Layer 3
        grad_z3 = grad_a3 * swish_derivative(self.z3)
        grad_W3 = self.a2.reshape(-1, 1) @ grad_z3.reshape(1, -1)
        grad_b3 = grad_z3
        grad_a2 = grad_z3 @ self.W3.T

        # Layer 2
        grad_z2 = grad_a2 * swish_derivative(self.z2)
        grad_W2 = self.a1.reshape(-1, 1) @ grad_z2.reshape(1, -1)
        grad_b2 = grad_z2
        grad_a1 = grad_z2 @ self.W2.T

        # Layer 1
        grad_z1 = grad_a1 * swish_derivative(self.z1)
        grad_W1 = self.x.reshape(-1, 1) @ grad_z1.reshape(1, -1)
        grad_b1 = grad_z1

        # Update weights
        self.W4 -= lr * grad_W4
        self.b4 -= lr * grad_b4.flatten()
        self.W3 -= lr * grad_W3
        self.b3 -= lr * grad_b3
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def param_count(self):
        return (100*128 + 128 + 128*64 + 64 + 64*32 + 32 + 32*1 + 1)

    def save(self, path):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 W4=self.W4, b4=self.b4)

    def save_bin(self, path):
        """Save as flat binary for Go."""
        with open(path, 'wb') as f:
            for arr in [self.W1, self.b1, self.W2, self.b2,
                       self.W3, self.b3, self.W4, self.b4]:
                f.write(arr.astype(np.float32).tobytes())

# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVER MLP: 207→128→64→100
# ═══════════════════════════════════════════════════════════════════════════════

class ObserverMLP:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Input: 100 resonances + 6 chambers + 1 iterations + 100 fingerprint = 207
        self.W1 = np.random.randn(207, 128) * np.sqrt(2.0 / 207)
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, 100) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(100)

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = swish(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = swish(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3  # logits

    def backward(self, grad_out, lr=0.01):
        grad_z3 = grad_out
        grad_W3 = self.a2.reshape(-1, 1) @ grad_z3.reshape(1, -1)
        grad_b3 = grad_z3
        grad_a2 = grad_z3 @ self.W3.T

        grad_z2 = grad_a2 * swish_derivative(self.z2)
        grad_W2 = self.a1.reshape(-1, 1) @ grad_z2.reshape(1, -1)
        grad_b2 = grad_z2
        grad_a1 = grad_z2 @ self.W2.T

        grad_z1 = grad_a1 * swish_derivative(self.z1)
        grad_W1 = self.x.reshape(-1, 1) @ grad_z1.reshape(1, -1)
        grad_b1 = grad_z1

        self.W3 -= lr * grad_W3
        self.b3 -= lr * grad_b3
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def param_count(self):
        return (207*128 + 128 + 128*64 + 64 + 64*100 + 100)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def save_bin(self, path):
        with open(path, 'wb') as f:
            for arr in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
                f.write(arr.astype(np.float32).tobytes())

# ═══════════════════════════════════════════════════════════════════════════════
# RESONANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_resonance(text):
    """Compute 100D resonance vector from text."""
    text_lower = text.lower()
    resonances = np.zeros(100)

    for i, anchor in enumerate(ALL_ANCHORS[:100]):
        if anchor in text_lower:
            # Direct match
            resonances[i] = 1.0
        else:
            # Partial match (substring)
            for word in text_lower.split():
                if anchor in word or word in anchor:
                    resonances[i] = 0.5
                    break

    # Normalize
    norm = np.linalg.norm(resonances)
    if norm > 0:
        resonances = resonances / norm

    return resonances

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_training_data(n_samples=1000):
    """Generate synthetic training examples."""
    examples = []

    templates = {
        "FEAR": ["I'm terrified", "fear grips me", "so scared", "panic attack", "anxious about"],
        "LOVE": ["I love you", "warmth fills me", "tender moment", "deep affection", "cherish"],
        "RAGE": ["I'm furious", "rage inside", "hatred burns", "so angry", "bitter resentment"],
        "VOID": ["feel empty", "numb inside", "nothing matters", "hollow feeling", "detached"],
        "FLOW": ["curious about", "wonder what", "uncertain", "in flux", "transitioning"],
        "COMPLEX": ["ashamed of", "guilty feeling", "nostalgic", "bittersweet", "regret"]
    }

    for _ in range(n_samples):
        chamber = np.random.choice(CHAMBER_NAMES)
        template = np.random.choice(templates[chamber])

        # Add some noise words
        noise = np.random.choice(["today", "always", "never", "sometimes", "really", "very"])
        text = f"{template} {noise}"

        examples.append({
            "text": text,
            "chamber": chamber
        })

    return examples

def train_cloud(epochs=100, lr=0.01):
    """Train Cloud 200K."""
    print("=" * 60)
    print("  CLOUD 200K Training")
    print("=" * 60)

    # Initialize
    chambers = {name: ChamberMLP(seed=i) for i, name in enumerate(CHAMBER_NAMES)}
    observer = ObserverMLP(seed=42)

    total_params = sum(c.param_count() for c in chambers.values()) + observer.param_count()
    print(f"\nTotal parameters: {total_params:,} ({total_params/1000:.1f}K)")
    print(f"  Chambers: {sum(c.param_count() for c in chambers.values()):,}")
    print(f"  Observer: {observer.param_count():,}")

    # Generate data
    print("\nGenerating training data...")
    examples = generate_training_data(2000)
    print(f"  {len(examples)} examples")

    # Training loop
    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        np.random.shuffle(examples)

        for ex in examples:
            text = ex["text"]
            target_chamber = ex["chamber"]

            # Forward pass
            resonances = compute_resonance(text)

            # Chambers
            activations = {}
            for name, chamber in chambers.items():
                activations[name] = chamber.forward(resonances)

            # Compute loss (MSE)
            loss = 0.0
            for name in CHAMBER_NAMES:
                target = 1.0 if name == target_chamber else 0.0
                pred = activations[name]
                loss += (pred - target) ** 2

                # Backprop
                grad = 2 * (pred - target)
                chambers[name].backward(np.array([grad]), lr)

            total_loss += loss

            # Check accuracy
            pred_chamber = max(activations, key=activations.get)
            if pred_chamber == target_chamber:
                correct += 1

        avg_loss = total_loss / len(examples)
        accuracy = correct / len(examples)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={accuracy:.2%}")

    print("\nTraining complete!")

    return chambers, observer

def save_weights(chambers, observer, out_dir):
    """Save weights in both NPZ and BIN formats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving weights to {out_dir}/")

    for name, chamber in chambers.items():
        # NPZ for Python
        chamber.save(out_dir / f"chamber_{name.lower()}.npz")
        # BIN for Go
        chamber.save_bin(out_dir / f"chamber_{name.lower()}.bin")
        print(f"  chamber_{name.lower()}: {chamber.param_count():,} params")

    observer.save(out_dir / "observer.npz")
    observer.save_bin(out_dir / "observer.bin")
    print(f"  observer: {observer.param_count():,} params")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    chambers, observer = train_cloud(epochs=100, lr=0.01)

    out_dir = "/Users/ataeff/Downloads/arianna.c/weights/cloud"
    save_weights(chambers, observer, out_dir)

    # Verify
    print("\nVerifying saved weights...")
    import os
    total_size = 0
    for f in os.listdir(out_dir):
        if f.endswith('.bin'):
            size = os.path.getsize(os.path.join(out_dir, f))
            total_size += size
            print(f"  {f}: {size:,} bytes")
    print(f"  Total: {total_size:,} bytes ({total_size/1024:.1f} KB)")

    expected_params = 6 * 23297 + 41060  # ~181K
    print(f"\nExpected ~{expected_params:,} params ({expected_params*4/1024:.1f} KB)")
