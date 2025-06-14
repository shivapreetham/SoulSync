import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Optional: Customize voice properties
engine.setProperty('rate', 150)  # Speed (words per minute)
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Get user input
text = input("Enter the text to convert to speech: ")

# Speak the text
engine.say(text)
engine.runAndWait()

# Optionally save to file
engine.save_to_file(text, 'output.mp3')
engine.runAndWait()

# Play the saved file (optional)
import os
os.system("start output.mp3")