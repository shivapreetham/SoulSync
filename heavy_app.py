import gradio as gr
import torch
from llm_utils2 import load_model, generate_response, get_smart_fallback
from chat_utils2 import build_prompt, truncate_history, validate_response, detect_conversation_quality

# Load model at startup (hardcoded for simplicity)
model_name = "facebook/blenderbot-3B"
try:
    tokenizer, model, device = load_model(model_name)
except Exception as e:
    raise Exception(f"Model loading failed: {str(e)}")

def detect_emotion_from_text(text: str) -> str:
    """
    Simple emotion detection based on keywords and patterns
    """
    text_lower = text.lower()
    
    # Emotion keyword mapping
    emotion_keywords = {
        "happy": ["happy", "joy", "excited", "great", "awesome", "wonderful", "amazing", "love", "fantastic"],
        "sad": ["sad", "depressed", "down", "upset", "hurt", "disappointed", "cry", "awful", "terrible"],
        "angry": ["angry", "mad", "furious", "annoyed", "irritated", "hate", "frustrated", "pissed"],
        "frustrated": ["frustrated", "stuck", "annoying", "difficult", "struggling", "can't", "won't work"],
        "confused": ["confused", "don't understand", "unclear", "puzzled", "lost", "what", "how", "why"],
        "excited": ["excited", "can't wait", "amazing", "incredible", "wow", "omg", "awesome", "fantastic"],
        "anxious": ["worried", "nervous", "anxious", "scared", "afraid", "concerned", "stress", "panic"],
        "tired": ["tired", "exhausted", "sleepy", "worn out", "drained", "fatigue"]
    }
    
    # Punctuation patterns
    if text.count('!') >= 2:
        return "excited"
    elif text.count('?') >= 2:
        return "confused"
    elif text.isupper() and len(text) > 10:
        return "angry"
    
    # Keyword matching
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    
    return "neutral"

def chat_function(user_input, history, temperature, top_k, top_p, max_tokens):
    """Handle user input and generate bot response"""
    if not user_input:
        return history, history
    
    # Detect emotion
    emotion = detect_emotion_from_text(user_input)
    
    # Prepare conversation history for prompt
    conversation_history = [(entry["user_input"], entry["bot_response"]) for entry in history]
    prompt = build_prompt(conversation_history, user_input, emotion, model_name, tokenizer)
    
    # Generate response
    response = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        history=conversation_history,
        user_input=user_input,
        emotion=emotion
    )
    
    # Validate response and use fallback if needed
    is_valid, final_response = validate_response(response, user_input, emotion)
    if not is_valid:
        final_response = get_smart_fallback(user_input, emotion)
    
    # Update history
    new_entry = {"user_input": user_input, "bot_response": final_response, "emotion": emotion}
    updated_history = history + [new_entry]
    
    # Truncate history if needed
    truncated_history = truncate_history(updated_history, tokenizer, max_tokens=800)
    
    # Format chatbot display with emojis
    chatbot_history = []
    for entry in truncated_history:
        emotion_emoji = {
            "happy": "😊", "sad": "😢", "angry": "😠", "frustrated": "😤",
            "confused": "🤔", "excited": "🤩", "anxious": "😰", "tired": "😴"
        }.get(entry["emotion"], "💬")
        user_message = f"{emotion_emoji} {entry['user_input']}"
        chatbot_history.append({"role": "user", "content": user_message})
        chatbot_history.append({"role": "assistant", "content": entry['bot_response']})
    
    return chatbot_history, truncated_history

def update_stats(history):
    """Update session statistics"""
    if not history:
        return "0", "None", "Unknown"
    
    num_exchanges = len(history)
    emotions = [entry["emotion"] for entry in history]
    most_common_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"
    quality_metrics = detect_conversation_quality([(entry["user_input"], entry["bot_response"]) for entry in history])
    quality = quality_metrics.get("quality", "unknown")
    
    return str(num_exchanges), most_common_emotion.title(), quality.title()

def clear_chat():
    """Clear the chat and reset stats"""
    return [], [], "0", "None", "Unknown"

def export_chat(history):
    """Export chat history to a file"""
    if not history:
        return None
    
    chat_text = "\n".join([
        f"User: {entry['user_input']}\nBot: {entry['bot_response']}\nEmotion: {entry['emotion']}\n---"
        for entry in history
    ])
    
    with open("soul_sync_chat.txt", "w") as f:
        f.write(chat_text)
    
    return "soul_sync_chat.txt"

# Define the Gradio interface
with gr.Blocks(title="Soul Sync Chat") as demo:
    gr.Markdown("# 🤖 Soul Sync Chat")
    gr.Markdown("*An advanced emotion-aware chatbot for meaningful conversations*")
    
    with gr.Row():
        # Sidebar-like column for controls and stats
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            temperature = gr.Slider(label="Temperature", minimum=0.6, maximum=1.2, value=0.8, step=0.1,
                                   info="Higher = more creative, Lower = more focused")
            top_k = gr.Slider(label="Top-k", minimum=30, maximum=100, value=50, step=10,
                             info="Number of top tokens to consider")
            top_p = gr.Slider(label="Top-p", minimum=0.7, maximum=0.95, value=0.9, step=0.05,
                             info="Nucleus sampling threshold")
            max_tokens = gr.Number(label="Max tokens", value=50, minimum=20, maximum=100, step=10,
                                  info="Maximum response length")
            
            gr.Markdown("### 📊 Session Stats")
            num_messages = gr.Textbox(label="Messages", value="0", interactive=False)
            dominant_emotion = gr.Textbox(label="Dominant Emotion", value="None", interactive=False)
            conversation_quality = gr.Textbox(label="Conversation Quality", value="Unknown", interactive=False)
            
            gr.Markdown("### 🔧 Controls")
            clear_button = gr.Button("🗑️ Clear Chat")
            export_button = gr.Button("💾 Export Chat")
            file_output = gr.File(label="Download Chat", type="filepath")
        
        # Main chat column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", label="💬 Conversation")
            user_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            submit_button = gr.Button("Send")
    
    # State to store conversation history
    history_state = gr.State([])
    
    # Event handlers
    submit_button.click(
        fn=chat_function,
        inputs=[user_input, history_state, temperature, top_k, top_p, max_tokens],
        outputs=[chatbot, history_state]
    ).then(
        fn=update_stats,
        inputs=[history_state],
        outputs=[num_messages, dominant_emotion, conversation_quality]
    )
    
    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, history_state, num_messages, dominant_emotion, conversation_quality]
    )
    
    export_button.click(
        fn=export_chat,
        inputs=[history_state],
        outputs=[file_output]
    )

# Launch the app
demo.launch()