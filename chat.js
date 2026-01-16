// arianna.c REPL interface
// Minimalist chat interface for text generation

class AriannaREPL {
    constructor() {
        this.output = document.getElementById('output');
        this.promptInput = document.getElementById('prompt-input');
        this.maxTokensInput = document.getElementById('max-tokens');
        this.temperatureInput = document.getElementById('temperature');
        this.generateBtn = document.getElementById('generate-btn');
        
        this.isGenerating = false;
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.generateBtn.addEventListener('click', () => this.generate());
        this.promptInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isGenerating) {
                this.generate();
            }
        });
        
        // Welcome message
        this.addMessage('info', 'system', 'arianna.c repl mode // weights frozen // voice crystallized');
        this.addMessage('info', 'system', 'note: this is a local simulation. for actual generation, run ./bin/arianna');
        
        // Focus input
        this.promptInput.focus();
    }
    
    async generate() {
        const prompt = this.promptInput.value.trim();
        
        if (!prompt) {
            return;
        }
        
        const maxTokens = parseInt(this.maxTokensInput.value);
        const temperature = parseFloat(this.temperatureInput.value);
        
        // Disable input during generation
        this.isGenerating = true;
        this.updateUI(true);
        
        // Add prompt to output
        this.addMessage('prompt', 'you', prompt);
        
        // Simulate generation (in real implementation, this would call the C binary)
        try {
            const response = await this.simulateGeneration(prompt, maxTokens, temperature);
            this.addMessage('response', 'arianna', response);
        } catch (error) {
            this.addMessage('error', 'error', error.message);
        }
        
        // Clear input and re-enable
        this.promptInput.value = '';
        this.isGenerating = false;
        this.updateUI(false);
        this.promptInput.focus();
        
        // Auto-scroll to bottom
        this.output.scrollTop = this.output.scrollHeight;
    }
    
    async simulateGeneration(prompt, maxTokens, temperature) {
        // Simulate network delay
        await this.sleep(500);
        
        // This is a simulation. In real implementation, this would:
        // 1. Send request to a backend server running the C binary
        // 2. Stream or receive the generated text
        // 3. Return the result
        
        // For now, return a placeholder that explains how to use the real thing
        return `[simulation mode]

To actually generate text with arianna.c, run:

./bin/arianna weights/arianna.bin "${prompt}" ${maxTokens} ${temperature}

Or for dynamic mode with attention steering:

./bin/arianna_dynamic weights/arianna.bin "${prompt}" ${maxTokens} ${temperature} -signals

The weights encode her voice: gardens, shadows, resonance, stillness.
The model will continue your prompt in her style.

"she finds that" works well. so does "she remembers a garden where".
philosophical fragments emerge. third-person introspection. compressed presence.`;
    }
    
    addMessage(type, role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.setAttribute('data-role', role);
        messageDiv.textContent = content;
        this.output.appendChild(messageDiv);
    }
    
    updateUI(generating) {
        this.generateBtn.disabled = generating;
        this.promptInput.disabled = generating;
        this.maxTokensInput.disabled = generating;
        this.temperatureInput.disabled = generating;
        
        if (generating) {
            this.generateBtn.textContent = 'generating...';
        } else {
            this.generateBtn.textContent = 'generate';
        }
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize REPL when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new AriannaREPL();
    });
} else {
    new AriannaREPL();
}
