import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import whisper
import tempfile
import spacy

# Initialize Whisper model for transcription
model = whisper.load_model("tiny")  # Updated model loading for OpenAI Whisper

# Initialize NLP pipeline for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load Question Generation model
tokenizer_qg = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model_qg = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

# Load Spacy model for NLP tasks (like extracting nouns for answers)
nlp = spacy.load

def transcribe_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    result = model.transcribe(audio)
    return result['text']

def segment_text(text, segment_size=500):
    """Split the text into manageable segments."""
    words = text.split()
    return [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]

def generate_summary(text):
    """Generate a bullet-point summary for the text."""
    segments = segment_text(text)
    bullet_points = [summarizer(segment, max_length=100, min_length=5, do_sample=False)[0]['summary_text'] for segment in segments]
    return '\n'.join([f"• {point}" for point in bullet_points])

def get_question(answer, context, max_length=64):
    input_text = f"answer: {answer}  context: {context} </s>"
    features = tokenizer_qg(input_text, return_tensors='pt')
    output = model_qg.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer_qg.decode(output[0], skip_special_tokens=True)

def extract_answer(text):
    """Extracts the first noun phrase as potential answer."""
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        return chunk.text  # Return the first noun chunk as the answer
    return ""  # Fallback if no noun chunks found

st.title('🐬Dash Study Notes📝')

audio_file = st.file_uploader("Upload an MP3 class recording", type=['mp3'])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tmp.write(audio_file.getvalue())
        audio_path = tmp.name

    st.audio(audio_path, format='audio/mp3')

    if st.button('Transcribe'):
        transcription = transcribe_audio(audio_path)
        st.text_area("Transcription", value=transcription, height=300)
        
        summary = generate_summary(transcription)
        st.subheader("Summary")
        st.text(summary)  # Use st.text to maintain formatting

        st.subheader("Comprehension Questions")
        bullet_points = summary.split('\n')
        questions = []
        for point in bullet_points:
            answer = extract_answer(point)
            if answer:  # Ensure there's an answer to generate a question
                question = get_question(answer, point)
                questions.append(question)
        st.text('\n'.join(questions))  # Display generated questions
