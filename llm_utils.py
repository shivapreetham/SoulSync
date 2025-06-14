import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import random

def load_model(model_name="microsoft/DialoGPT-large", device=None):
    """
    Load model with optimized configuration for better conversational models
    """
    print(f"Loading {model_name}...")
    
    # Configure quantization for larger models
    quantization_config = None
    if torch.cuda.is_available() and "3b" in model_name.lower():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with appropriate settings
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available() and not quantization_config:
        model.to(device)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return tokenizer, model, device


def clean_response(text: str, user_input: str = "") -> str:
    """
    Enhanced response cleaning with better context awareness
    """
    if not text:
        return ""
    
    # Remove common artifacts and stop sequences
    stop_sequences = [
        "Human:", "Assistant:", "User:", "AI:", "Bot:", "Person:",
        "\nHuman", "\nUser", "\nAssistant", "\nAI", "\nBot",
        "<|endoftext|>", "<|end|>", "</s>", "<s>", "[INST]", "[/INST]"
    ]
    
    for seq in stop_sequences:
        if seq in text:
            text = text.split(seq)[0]
    
    # Remove repetitive patterns
    text = re.sub(r'(.{10,}?)\1{2,}', r'\1', text)  # Remove long repetitions
    text = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', text)  # Remove triple word repetitions
    text = re.sub(r'(\w+\s+){5,}\1', r'\1', text)  # Remove pattern loops
    
    # Clean up formatting
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[.,!?;:]+', '', text).strip()
    text = re.sub(r'[.,!?;:]{3,}', '.', text)
    
    # Remove filler starts
    text = re.sub(r'^(um+|uh+|er+|ah+|well|so|like|okay)\s+', '', text, flags=re.IGNORECASE)
    
    # Handle sentence completion
    if text and not text[-1] in '.!?':
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1 and len(sentences[0].strip()) > 15:
            text = sentences[0].strip() + '.'
        elif len(text) > 100:
            last_space = text[:100].rfind(' ')
            if last_space > 50:
                text = text[:last_space] + "."
    
    text = text.strip()
    
    # Quality filters
    if (len(text) < 5 or 
        text.count('haha') > 2 or
        text.lower().startswith(('umm', 'uh', 'what?', 'sorry, i')) or
        'i cannot' in text.lower() or
        'i am not sure' in text.lower()):
        return ""
    
    return text


def apply_emotion_context(prompt: str, emotion: str, user_input: str) -> str:
    """
    Apply emotion-aware context to prompts
    """
    if emotion == "neutral":
        return prompt
    
    emotion_contexts = {
        "happy": "The user is feeling happy and positive. Respond with enthusiasm and share in their joy.",
        "excited": "The user is excited about something. Match their energy level appropriately.",
        "sad": "The user seems sad or down. Be empathetic, supportive, and understanding.",
        "frustrated": "The user appears frustrated. Be patient, helpful, and solution-focused.",
        "confused": "The user seems confused. Provide clear, simple explanations.",
        "angry": "The user seems upset or angry. Stay calm, be understanding, and de-escalate.",
        "anxious": "The user appears anxious. Be reassuring and supportive.",
        "tired": "The user seems tired. Be gentle and understanding."
    }
    
    if emotion in emotion_contexts:
        context = emotion_contexts[emotion]
        return f"{context}\n\n{prompt}"
    
    return prompt


def generate_response(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    history=None,
    user_input: str = "",
    emotion: str = "neutral"
) -> str:
    """
    Generate response with better parameters for conversational models
    """
    
    try:
        # Apply emotion context
        prompt = apply_emotion_context(prompt, emotion, user_input)
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,  # Increased for better context
            padding=True,
            return_attention_mask=True
        )
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        seq_len = input_ids.shape[-1]
        max_length = min(seq_len + max_new_tokens, 1200)
        
        # Generate with better parameters
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=seq_len + 10,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True,
                use_cache=True
            )
        
        # Decode new tokens
        new_tokens = output[0][seq_len:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean response
        cleaned_response = clean_response(new_text, user_input)
        
        return cleaned_response
        
    except Exception as e:
        print(f"Generation error: {e}")
        return ""


def get_smart_fallback(user_input: str, emotion: str = "neutral") -> str:
    """
    Intelligent fallback responses based on user input and emotion
    """
    user_lower = user_input.lower().strip()
    
    # Factual question patterns
    factual_patterns = {
        r"capital.*india": "The capital of India is New Delhi.",
        r"president.*usa|usa.*president": "The current US President is Donald Trump (as of 2025).",
        r"past.*form.*do": "The past form of 'do' is 'did'.",
        r"present.*form.*did": "The present form of 'did' is 'do'.",
        r"what.*is.*2\+2|2\+2": "2 + 2 equals 4.",
        r"largest.*country": "Russia is the largest country by land area.",
        r"smallest.*country": "Vatican City is the smallest country in the world."
    }
    
    for pattern, response in factual_patterns.items():
        if re.search(pattern, user_lower):
            return response
    
    # Greeting patterns
    if re.search(r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b', user_lower):
        greetings = [
            "Hello! How are you doing today?",
            "Hi there! What's on your mind?",
            "Hey! How can I help you today?",
            "Hello! I'm here to chat with you."
        ]
        return random.choice(greetings)
    
    # Emotion-based responses
    emotion_responses = {
        "happy": [
            "That's wonderful! I'm glad you're feeling good. What's making you happy?",
            "Your happiness is contagious! Tell me more about what's going well.",
            "I love hearing when people are happy! What's the source of your joy?"
        ],
        "sad": [
            "I'm sorry you're feeling down. Would you like to talk about what's bothering you?",
            "It's okay to feel sad sometimes. I'm here to listen if you need to share.",
            "I understand you're going through a tough time. How can I help?"
        ],
        "frustrated": [
            "I can sense your frustration. What's causing you stress right now?",
            "Frustration is tough to deal with. Want to talk about what's bothering you?",
            "I understand you're feeling frustrated. Let's work through this together."
        ],
        "confused": [
            "It's perfectly normal to feel confused sometimes. What can I help clarify?",
            "I'm here to help clear things up. What's puzzling you?",
            "Confusion happens to everyone. What would you like me to explain?"
        ],
        "angry": [
            "I can tell you're upset. Take your time, and let me know how I can help.",
            "It's okay to feel angry. Would you like to talk about what's bothering you?",
            "I understand you're angry. I'm here to listen without judgment."
        ],
        "excited": [
            "Your excitement is wonderful! What's got you so energized?",
            "I love your enthusiasm! Tell me what's making you so excited.",
            "Your energy is amazing! What's the exciting news?"
        ]
    }
    
    if emotion in emotion_responses:
        return random.choice(emotion_responses[emotion])
    
    # Question patterns
    if re.search(r'\b(what|who|when|where|why|how)\b', user_lower):
        return "That's a great question! I'd be happy to help you with that. Can you be more specific?"
    
    # Help patterns
    if re.search(r'\b(help|assist|support)\b', user_lower):
        return "I'm here to help! What do you need assistance with?"
    
    # Default contextual responses
    defaults = [
        "That's interesting! Tell me more about that.",
        "I'm listening. What would you like to discuss?",
        "I'm here to chat. What's on your mind?",
        "That sounds important to you. Can you elaborate?"
    ]
    
    return random.choice(defaults)