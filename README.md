This project implements an automated pipeline for audio processing and music information retrieval. The system follows a modular flow from user input to feature-driven results.

**LINK:**  https://musicgenreclassification-gvunx8qtz6cpkx4sslsebp.streamlit.app/

**System Workflow**

User Authentication: Secure entry point via User Login to manage personal audio libraries.

Data Ingestion: Users Upload Audio files in supported formats (e.g., WAV, MP3).

Preprocessing: Initial signal processing, including noise reduction, normalization, and trimming.

Feature Scaling: Standardizing data ranges (e.g., using Min-Max Scaling or Z-score normalization) to ensure optimal model performance.

Feature Extraction: Leveraging libraries like Librosa or Essentia to extract spectral and temporal characteristics.

Analysis Streams:
MLP Genre Classification: A Multi-Layer Perceptron (MLP) neural network categorizes the audio into specific musical genres.
Music Information Retrieval (MIR): Parallel extraction of technical metadata including Pitch, Rhythm, Scale, Metronome (BPM), and Chords.

Output & Persistence:
Result Display: An interactive UI presents the classification and musical attributes to the user.
Save to Database: All extracted features and results are archived for future reference and historical analysis.

**Technical Stack**

Machine Learning: Multi-Layer Perceptron (MLP)

Audio Processing: Librosa 

Dataset : GTZAN
