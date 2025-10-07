import streamlit as st
import os
import tempfile
import numpy as np
import time
import scipy.io.wavfile as wavfile
import math
import pygame # Add pygame for MIDI playback

# Define the music generation/remixing logic function
# This is a placeholder implementation. In a real application,
# this would involve loading and running an AI model (e.g., Magenta).
def generate_music(uploaded_file_path, mood_genre, status_placeholder):
    """
    Placeholder function for AI music generation/remixing with progress updates.

    Args:
        uploaded_file_path (str): The path to the uploaded audio file.
        mood_genre (str): The selected mood or genre.
        status_placeholder: Streamlit empty object to display status messages.

    Returns:
        str: The path to the generated audio file.
    """
    status_placeholder.text("Processing audio...")
    time.sleep(1) # Simulate processing time

    status_placeholder.text(f"Generating music based on {os.path.basename(uploaded_file_path)} with mood/genre: {mood_genre}...")
    time.sleep(2) # Simulate generation time

    # --- Placeholder for actual AI music generation logic ---
    # This part would involve:
    # 1. Loading the uploaded audio file.
    # 2. Processing the audio (e.g., converting to a different format, extracting features).
    # 3. Using an AI model (like Magenta) to remix or generate new music
    #    based on the input audio and the selected mood/genre.
    # 4. Synthesizing the model's output into an audio format.
    # ----------------------------------------------------------

    # For demonstration purposes, let's create a dummy piano-like tone
    sample_rate = 44100
    duration = 5 # seconds
    # Generate a simple sequence of notes (e.g., C4, E4, G4)
    midi_notes = [60, 64, 67, 72] # MIDI notes for C4, E4, G4, C5
    note_duration = duration / len(midi_notes)
    audio_data = np.array([], dtype=np.float32) # Use float32 for soundfile

    for midi_note in midi_notes:
        # Convert MIDI note to frequency
        frequency = 440 * (2 ** ((midi_note - 69) / 12))
        # Generate sine wave for the note
        t = np.linspace(0., note_duration, int(sample_rate * note_duration), endpoint=False)
        amplitude = 0.5 # Reduced amplitude
        note_audio = amplitude * np.sin(2. * np.pi * frequency * t)

        # Add a simple decay envelope (optional, for a more piano-like sound)
        decay_envelope = np.linspace(1, 0.1, len(note_audio))
        note_audio *= decay_envelope

        audio_data = np.append(audio_data, note_audio)

    # Create a temporary file to save the generated audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        output_filename = tmpfile.name
        wavfile.write(output_filename, sample_rate, audio_data) # Use scipy.io.wavfile to write

    status_placeholder.text("Music generation complete!")
    return output_filename

st.title("AI Music Remix and Generation")
st.write("Upload a song, select a mood or genre, and let AI create a remix or new music!")

uploaded_file = st.file_uploader("Upload your song (MP3, WAV, etc.)", type=["mp3", "wav", "ogg"])

mood_genre = st.selectbox(
    "Select a mood or genre",
    ["Happy", "Sad", "Energetic", "Relaxing", "Pop", "Rock", "Electronic", "Hip Hop"]
)

# Placeholder for progress/status message
status_placeholder = st.empty()
output_placeholder = st.empty()


if st.button("Remix/Generate Music"):
    if uploaded_file is not None:
        # File Validation
        file_size_limit_mb = 20
        if uploaded_file.size > file_size_limit_mb * 1024 * 1024:
            status_placeholder.error(f"Uploaded file is too large. Please upload a file smaller than {file_size_limit_mb} MB.")
        # Add other validation checks here if needed (e.g., more detailed format check)
        else:
            status_placeholder.text("Uploading file...")
            time.sleep(0.5) # Simulate upload time

            # Save the uploaded file to a temporary location to get a path
            uploaded_file_path = None # Initialize to None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    uploaded_file_path = tmp_file.name

                # Call the music generation function with status_placeholder
                generated_audio_path = generate_music(uploaded_file_path, mood_genre, status_placeholder)

                # Store the generated audio path in session state
                st.session_state['generated_audio_path'] = generated_audio_path

                status_placeholder.success("Music generation successful!") # Use success for clear feedback

            except FileNotFoundError:
                 status_placeholder.error("Error: Temporary file could not be created or accessed.")
            # Removed soundfile specific error handling
            except Exception as e:
                status_placeholder.error(f"An unexpected error occurred during music generation: {e}")

            finally:
                 # Clean up the temporary uploaded file if it was created
                if uploaded_file_path and os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path)

    else:
        status_placeholder.warning("Please upload an audio file first.")

st.subheader("Output Music")

# Display the audio player if generated audio is available in session state
if 'generated_audio_path' in st.session_state and os.path.exists(st.session_state['generated_audio_path']):
     # Read the generated audio file into bytes
    try:
        with open(st.session_state['generated_audio_path'], 'rb') as f:
            audio_bytes = f.read()

        output_placeholder.audio(audio_bytes, format='audio/wav')

        # Optional: Add a download button
        st.download_button(
            label="Download Generated Music",
            data=audio_bytes,
            file_name="generated_music.wav",
            mime="audio/wav"
        )
    except FileNotFoundError:
        output_placeholder.error("Error: Generated audio file not found.")
    except Exception as e:
        output_placeholder.error(f"An error occurred while loading the generated audio: {e}")

    finally:
        # Clean up the temporary generated file after it has been displayed/downloaded
        # This might need adjustment if the user needs to play it multiple times without regeneration
        # For now, clean up after display attempt.
        if 'generated_audio_path' in st.session_state and os.path.exists(st.session_state['generated_audio_path']):
             try:
                 # Keep the path in session state until the user triggers a new generation
                 # or the session ends, to allow playback.
                 pass
             except OSError as e:
                 print(f"Error deleting temporary generated file: {e}") # Print to console for debugging
