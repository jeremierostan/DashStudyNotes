# 🐬Dash Study Notes📝
*Transcribes, summarizes, and checks for understanding of class recordings*

## Introduction
This app is a devemopment of the original [DashNotes](https://github.com/jeremierostan/DashNotes). In addition to transcribing and summarizing voice recordings, this version generates comprehension questions to aid and check for understanding.

## Question Generation
The generation of questions is handled by [mrm8488/t5-base-finetuned-question-generation-ap](https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap), a Google T5-based model fine-tuned for question generation by Manuel Romero on the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/). After transcription through openai-whisper, [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) provides a bullet point summary, which is used as the "context" of the question-generation prompt, while the "answer" is extracted with spaCy through a simple part-of-speech tagging task of the first noun chunk. 

## Limitation
The process described above comes with many limitations, including:
- Potential transcription errors (less likely)
- Potential summarization errors (somewhat likely)
- Potentially irrelevant questions (more likely)
This all highlights the fact that, just like the original app, Dash Study Notes is an experimental proof-of-concept. By design, it is limited to open-source and no-cost resources. Similar functionalities and fasterm better performance can be obtained (quite easily) with paid options. 

## Installation
To run the app:
1) Download Python
2) Download a code editor
3) Create a virtual environment (optional)
4) Open dash_study_notes.py
5)Install the necessary Python packages:
```
pip install streamlit==1.10.0 transformers==4.20.1 openai-whisper==20231117 spacy==3.2.1
```

## Usage
1) Start the app:
```
streamlit run dash_study_notes.py
```
2) Upload an MP3 class recording through the app's interface.
3) Click the 'Transcribe' button to transcribe the audio.
4) The app will display the transcription, a generated summary, and comprehension questions based on the summary.

NB. You will have to be PATIENT!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
