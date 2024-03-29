import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import whisper
import tempfile
import spacy

@st.cache(allow_output_mutation=True)
def load_whisper_model():
    return whisper.load_model("tiny")

@st.cache(allow_output_mutation=True)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache(allow_output_mutation=True)
def load_question_generation_model():
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    return tokenizer, model

@st.cache(allow_output_mutation=True)
def load_spacy_model():
    return spacy.load("en_core_web_sm")

model = load_whisper_model()
summarizer = load_summarizer()
tokenizer_qg, model_qg = load_question_generation_model()
nlp = load_spacy_model()  

def transcribe_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    result = model.transcribe(audio)
    return result['text']

def segment_text(text, segment_size=500):
    words = text.split()
    return [' '.join(words[i:i+segment_size]) for i in range(0, len(words), segment_size)]

def generate_summary(text):
    segments = segment_text(text)
    bullet_points = [summarizer(segment, max_length=100, min_length=5, do_sample=False)[0]['summary_text'] for segment in segments]
    return '\n'.join([f"• {point}" for point in bullet_points])

def get_question(answer, context, max_length=64):
    input_text = f"answer: {answer} context: {context} </s>"
    features = tokenizer_qg(input_text, return_tensors='pt')
    output = model_qg.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    return tokenizer_qg.decode(output[0], skip_special_tokens=True)

def extract_answer(text):
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        return chunk.text  
    return ""  

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
        st.text(summary)  

        st.subheader("Comprehension Questions")
        bullet_points = summary.split('\n')
        questions = []
        for point in bullet_points:
            answer = extract_answer(point)
            if answer:  
                question = get_question(answer, point)
                questions.append(question)
        st.text('\n'.join(questions))  

