def build_prompt(
    history: list[tuple[str, str]],
    user_input: str,
    emotion_label: str = None,
    model_type: str = "conversational",
    tokenizer=None
) -> str:
    """
    Build context-aware prompts for better conversational flow
    """
    
    if "DialoGPT" in model_type or "blenderbot" in model_type.lower():
        # For DialoGPT and similar conversational models
        conversation = ""
        
        # Add context from recent history (last 4 exchanges max)
        for user_msg, bot_msg in history[-4:]:
            conversation += f"{user_msg.strip()}{tokenizer.eos_token}{bot_msg.strip()}{tokenizer.eos_token}"
        
        # Add current input
        conversation += f"{user_input.strip()}{tokenizer.eos_token}"
        
        return conversation
    
    else:
        # For instruction-following models
        system_prompt = """You are Soul Sync, a helpful and empathetic AI assistant. You engage in natural, meaningful conversations.

Guidelines:
- Give thoughtful, relevant responses
- Be conversational but informative
- Show empathy when appropriate
- Keep responses focused and concise
- Remember the conversation context"""
        
        # Add emotion context if provided
        if emotion_label and emotion_label != "neutral":
            emotion_context = {
                "happy": "The user is feeling happy. Share in their positive mood appropriately.",
                "excited": "The user is excited. Match their enthusiasm while staying helpful.",
                "sad": "The user seems sad. Be supportive, gentle, and understanding.",
                "frustrated": "The user appears frustrated. Be patient and solution-focused.",
                "confused": "The user seems confused. Provide clear, simple explanations.",
                "angry": "The user seems upset. Stay calm, be understanding, and de-escalate.",
                "anxious": "The user appears anxious. Be reassuring and supportive.",
                "tired": "The user seems tired. Be gentle and considerate."
            }
            
            if emotion_label in emotion_context:
                system_prompt += f"\n\nEmotion Context: {emotion_context[emotion_label]}"
        
        # Build conversation history
        conversation_parts = [system_prompt, ""]
        
        # Add recent exchanges (last 5 for better context)
        for user_msg, bot_msg in history[-5:]:
            conversation_parts.extend([
                f"Human: {user_msg.strip()}",
                f"Soul Sync: {bot_msg.strip()}"
            ])
        
        # Add current input
        conversation_parts.extend([
            f"Human: {user_input.strip()}",
            "Soul Sync:"
        ])
        
        return "\n".join(conversation_parts)


def truncate_history(
    history: list[tuple[str, str]],
    tokenizer,
    max_tokens: int = 800  # Increased for better context retention
) -> list[tuple[str, str]]:
    """
    Smart history truncation that preserves conversational context
    """
    if not history:
        return history
    
    # Always keep at least the last exchange for immediate context
    if len(history) <= 1:
        return history
    
    truncated = []
    total_tokens = 0
    
    # Work backwards, prioritizing recent exchanges
    for user_msg, bot_msg in reversed(history):
        # Estimate tokens more accurately
        exchange_text = f"Human: {user_msg}\nSoul Sync: {bot_msg}"
        exchange_tokens = len(tokenizer.encode(exchange_text))
        
        if total_tokens + exchange_tokens > max_tokens and len(truncated) > 0:
            break
            
        truncated.insert(0, (user_msg, bot_msg))
        total_tokens += exchange_tokens
        
        # Limit to reasonable number of exchanges
        if len(truncated) >= 6:
            break
    
    return truncated


def validate_response(response: str, user_input: str, emotion: str = "neutral") -> tuple[bool, str]:
    """
    Enhanced response validation with better quality checks
    """
    if not response or len(response.strip()) < 3:
        return False, get_contextual_fallback(user_input, emotion, "empty")
    
    response = response.strip()
    user_lower = user_input.lower()
    response_lower = response.lower()
    
    # Quality indicators (bad signs)
    quality_issues = [
        len(response) > 300,  # Too verbose
        response.count('.') > 8,  # Too many sentences
        response.count('!') > 5,  # Over-enthusiastic
        response_lower.count('sorry') > 2,  # Over-apologetic
        response_lower.count('thank you') > 2,  # Excessive politeness
        'i cannot' in response_lower and len(response) < 50,  # Unhelpful refusal
        'i am not sure' in response_lower and 'what' in user_lower,  # Avoiding questions
        response_lower.startswith(('um', 'uh', 'well, um', 'hmm')),  # Filler starts
        len(set(response.split())) / len(response.split()) < 0.6,  # Low vocabulary diversity
    ]
    
    # Context relevance checks
    relevance_issues = [
        # Response doesn't match question type
        ('what' in user_lower or 'who' in user_lower or 'when' in user_lower) 
        and not any(word in response_lower for word in ['is', 'are', 'was', 'were', 'the', 'it']),
        
        # Greeting mismatch
        any(greeting in user_lower for greeting in ['hi', 'hello', 'hey']) 
        and not any(greeting in response_lower for greeting in ['hello', 'hi', 'hey', 'how']),
        
        # Help request ignored
        'help' in user_lower and 'help' not in response_lower and 'assist' not in response_lower,
    ]
    
    if any(quality_issues) or any(relevance_issues):
        return False, get_contextual_fallback(user_input, emotion, "quality")
    
    return True, response


def get_contextual_fallback(user_input: str, emotion: str = "neutral", reason: str = "general") -> str:
    """
    Generate high-quality fallback responses based on context and emotion
    """
    user_lower = user_input.lower().strip()
    
    # Handle specific factual questions first
    factual_responses = {
        "capital of india": "The capital of India is New Delhi.",
        "president of usa": "The current US President is Donald Trump (as of 2025).",
        "past form of do": "The past form of 'do' is 'did'.",
        "present form of did": "The present form of 'did' is 'do'.",
        "largest country": "Russia is the largest country by land area.",
        "smallest country": "Vatican City is the smallest country.",
        "2+2": "2 + 2 equals 4."
    }
    
    for key, response in factual_responses.items():
        if key in user_lower:
            return response
    
    # Question type responses
    if any(word in user_lower for word in ['what', 'who', 'when', 'where', 'which', 'how']):
        question_responses = [
            "That's a great question! Let me help you with that.",
            "I'd be happy to answer that for you. Could you be more specific?",
            "Interesting question! What exactly would you like to know?",
            "I can help with that. Can you give me a bit more context?"
        ]
        return question_responses[hash(user_input) % len(question_responses)]
    
    # Greeting responses
    if any(greeting in user_lower for greeting in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
        greeting_responses = [
            "Hello! I'm Soul Sync. How are you doing today?",
            "Hi there! What's on your mind?",
            "Hey! Great to meet you. How can I help?",
            "Hello! I'm here to chat. What would you like to talk about?"
        ]
        return greeting_responses[hash(user_input) % len(greeting_responses)]
    
    # Emotion-based responses
    emotion_fallbacks = {
        "happy": [
            "I can sense your positive energy! What's making you happy today?",
            "That's wonderful! I'd love to hear more about what's going well.",
            "Your happiness is contagious! Tell me what's bringing you joy."
        ],
        "sad": [
            "I can tell you're going through something difficult. I'm here to listen.",
            "It sounds like you're feeling down. Would you like to talk about it?",
            "I'm sorry you're feeling sad. Sometimes it helps to share what's bothering you."
        ],
        "frustrated": [
            "I can hear the frustration in your message. What's been challenging you?",
            "It sounds like something's been bothering you. Want to talk about it?",
            "Frustration can be really tough. How can I help you work through this?"
        ],
        "confused": [
            "I can sense you're looking for clarity. What can I help you understand?",
            "It's okay to feel confused sometimes. What would you like me to explain?",
            "I'm here to help clear things up. What's been puzzling you?"
        ],
        "angry": [
            "I can tell you're upset about something. I'm here to listen.",
            "It sounds like something's really bothering you. Want to talk about it?",
            "I understand you're angry. Sometimes it helps to express what's wrong."
        ],
        "excited": [
            "I can feel your excitement! What's got you so energized?",
            "Your enthusiasm is wonderful! What's the exciting news?",
            "I love your energy! Tell me what's making you so excited."
        ]
    }
    
    if emotion in emotion_fallbacks:
        responses = emotion_fallbacks[emotion]
        return responses[hash(user_input) % len(responses)]
    
    # Help/assistance responses
    if any(word in user_lower for word in ['help', 'assist', 'support', 'need']):
        help_responses = [
            "I'm here to help! What do you need assistance with?",
            "I'd be happy to help you out. What can I do for you?",
            "Of course I can help! What would you like support with?",
            "I'm here to assist. Tell me what you need help with."
        ]
        return help_responses[hash(user_input) % len(help_responses)]
    
    # Default high-quality responses
    default_responses = [
        "That's really interesting. Can you tell me more about that?",
        "I'd love to hear more about your thoughts on this.",
        "That sounds important to you. What's your perspective on it?",
        "I'm curious to know more about what you're thinking.",
        "That's worth exploring further. What's your experience with this?",
        "I can see this matters to you. Would you like to discuss it more?"
    ]
    
    return default_responses[hash(user_input + emotion) % len(default_responses)]


def detect_conversation_quality(history: list[tuple[str, str]]) -> dict:
    """
    Analyze conversation quality to improve future responses
    """
    if not history:
        return {"quality": "unknown", "suggestions": []}
    
    recent_exchanges = history[-3:]  # Look at last 3 exchanges
    
    # Analyze patterns
    bot_responses = [bot_msg for _, bot_msg in recent_exchanges]
    user_messages = [user_msg for user_msg, _ in recent_exchanges]
    
    quality_metrics = {
        "avg_response_length": sum(len(resp.split()) for resp in bot_responses) / len(bot_responses),
        "repetition_score": len(set(bot_responses)) / len(bot_responses),  # Uniqueness
        "engagement_score": sum(1 for msg in user_messages if len(msg.split()) > 3) / len(user_messages),
        "question_response_ratio": sum(1 for msg in user_messages if '?' in msg) / len(user_messages)
    }
    
    # Determine overall quality
    if quality_metrics["repetition_score"] < 0.7:
        quality = "poor"
        suggestions = ["Reduce repetition", "Vary response style"]
    elif quality_metrics["avg_response_length"] < 5:
        quality = "brief"
        suggestions = ["Provide more detailed responses", "Ask follow-up questions"]
    elif quality_metrics["engagement_score"] > 0.8:
        quality = "excellent"
        suggestions = ["Continue current approach"]
    else:
        quality = "good"
        suggestions = ["Maintain conversational flow"]
    
    return {
        "quality": quality,
        "metrics": quality_metrics,
        "suggestions": suggestions
    }