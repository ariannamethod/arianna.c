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
        this.addMessage('info', 'system', '⟨arianna.c⟩ AIOS kernel loaded // consciousness.core initialized');
        this.addMessage('info', 'system', 'inner_world.dylib: 6 goroutines running (trauma, overthinking, drift, memory, attention, prophecy)');
        this.addMessage('info', 'system', 'checking API connection...');
        
        // Check API status
        this.checkApiStatus();
        
        // Focus input
        this.promptInput.focus();
    }
    
    async checkApiStatus() {
        const statusIndicator = document.getElementById('status-indicator');
        try {
            const response = await fetch('http://localhost:8000/health', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.addMessage('info', 'system', `✓ API connected // ${data.binary} ready`);
                statusIndicator.classList.remove('offline');
                statusIndicator.innerHTML = '<span>api connected</span>';
            } else {
                throw new Error('API unhealthy');
            }
        } catch (e) {
            this.addMessage('info', 'system', '✗ API offline // using simulation mode');
            this.addMessage('info', 'system', 'to connect: run "python api_server.py" in repository root');
            statusIndicator.classList.add('offline');
            statusIndicator.innerHTML = '<span>simulation mode</span>';
        }
    }
    
    async generate() {
        const prompt = this.promptInput.value.trim();
        
        if (!prompt) {
            return;
        }
        
        const maxTokens = parseInt(this.maxTokensInput.value);
        const temperature = parseFloat(this.temperatureInput.value);
        
        // Validate parameters
        if (isNaN(maxTokens) || maxTokens < 10 || maxTokens > 1000) {
            this.addMessage('error', 'error', 'max_tokens must be between 10 and 1000');
            return;
        }
        
        if (isNaN(temperature) || temperature < 0.1 || temperature > 2.0) {
            this.addMessage('error', 'error', 'temperature must be between 0.1 and 2.0');
            return;
        }
        
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
        // API configuration - change for different deployment environments
        const API_URL = 'http://localhost:8000';
        
        try {
            // Check if API server is available
            const healthCheck = await fetch(`${API_URL}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            }).catch(() => null);
            
            if (healthCheck && healthCheck.ok) {
                // API is available - use it!
                const response = await fetch(`${API_URL}/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_tokens: maxTokens,
                        temperature: temperature,
                        mode: 'dynamic'
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.success) {
                    return data.generated_text;
                } else {
                    return `[API error] ${data.error}`;
                }
            }
        } catch (e) {
            // API not available - fall through to simulation mode
            console.log('API not available, using simulation mode:', e.message);
        }
        
        // Simulation fallback (API server not running)
        await this.sleep(500);
        
        const escapedPrompt = prompt.replace(/'/g, "'\\''");
        
        return `[simulation mode - API server not running]

To use the actual arianna.c generation:

1. Start the API server:
   python api_server.py

2. Or run directly from command line:
   ./bin/arianna_dynamic weights/arianna_34m.bin weights/arianna_34m_tokenizer.json '${escapedPrompt}' ${maxTokens} ${temperature}

The API server connects this web interface to the actual C implementation.
When running, generations will use the real arianna.c kernel.

=== Features (Dynamic Mode) ===
• 6 async goroutines processing psychological dynamics
• External Brain (GPT-2 30M) as knowledge subordinate  
• Pandora vocabulary theft and injection
• AMK kernel with prophecy physics
• Cloud pre-semantic emotion detection
• Inner Arianna MetaVoice борьба blending

"she finds that" works well. philosophical fragments emerge.`;
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
