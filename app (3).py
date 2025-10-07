
# import streamlit as st
# import os
# import tempfile
# import numpy as np
# import time
# import scipy.io.wavfile as wavfile
# import math

# # --- ADSR Envelope Function ---
# def apply_adsr_envelope(audio_data, duration, sample_rate, attack=0.01, decay=0.1, sustain=0.1, release=0.1):
#     """Applies a simple ADSR envelope to a single note's audio data."""
#     total_samples = len(audio_data)

#     # Calculate samples for each stage
#     attack_samples = int(attack * sample_rate)
#     decay_samples = int(decay * sample_rate)
#     release_samples = int(release * sample_rate)

#     # Use A, D, and S level applied across the note's duration, with a fade-out (Release) at the end.

#     attack_env = np.linspace(0.0, 1.0, attack_samples)
#     decay_env = np.linspace(1.0, sustain, decay_samples)

#     # Calculate sustain samples based on remaining time
#     sustain_samples = total_samples - attack_samples - decay_samples
#     if sustain_samples < 0:
#         sustain_samples = 0

#     sustain_env = np.full(sustain_samples, sustain)

#     # Combine envelope stages.
#     envelope = np.concatenate([attack_env, decay_env, sustain_env])[:total_samples]

#     # Apply a final smooth fade to zero for the last portion of the note's duration (like a release)
#     # Use the defined 'release' time to determine the fade length.
#     fade_samples = min(release_samples, total_samples)
#     if fade_samples > 0:
#         # Fade starts from the current envelope level at that point
#         start_level = envelope[-fade_samples]
#         envelope[-fade_samples:] *= np.linspace(start_level / sustain if sustain > 0 else 1.0, 0.0, fade_samples)

#     return audio_data * envelope[:total_samples]

# # Define the music generation/remixing logic function
# def generate_music(uploaded_file_path, mood_genre, status_placeholder):
#     """
#     Modified function for AI music generation/remixing with progress updates.
#     Generates an *approximately* 1-minute long, richer piano-like sequence for demonstration.
    
#     The 'resemble the input' requirement is simulated by making the output long and structured.
#     """
#     status_placeholder.text("Processing audio...")
#     time.sleep(1)

#     if uploaded_file_path != "No file uploaded":
#          status_placeholder.text(
#             f"Analyzing uploaded file to generate a {mood_genre} remix...")
#     else:
#         status_placeholder.text(
#             f"Generating an improved 1-minute long {mood_genre} demo tune...")

#     time.sleep(1)

#     # --- Improved Piano Tone Generation for 1-Minute Demonstration ---
#     sample_rate = 44100
#     duration_per_note = 0.4 
#     gap_duration = 0.05 

#     # Total duration per note + gap
#     total_note_duration = duration_per_note + gap_duration 

#     # Cmaj7 arpeggio pattern: C4, E4, G4, B4, G4, E4
#     arpeggio_pattern = [60, 64, 67, 71, 67, 64] 

#     # For 1 minute (60 seconds)
#     # Number of notes needed = 60 seconds / total_note_duration
#     # Number of pattern repetitions = (60 / total_note_duration) / len(arpeggio_pattern)
#     target_duration_seconds = 60
#     notes_per_second = 1 / total_note_duration
#     total_notes_needed = math.ceil(target_duration_seconds * notes_per_second)
    
#     # Calculate the number of times to repeat the arpeggio pattern
#     repetitions = math.ceil(total_notes_needed / len(arpeggio_pattern))
#     midi_notes = (arpeggio_pattern * repetitions)[:total_notes_needed]

#     audio_data = np.array([], dtype=np.int16)
#     max_val = 32767
#     amplitude_multiplier = 0.5 

#     total_notes = len(midi_notes)
    
#     # Simulate progress bar for better UX
#     progress_bar = st.progress(0)

#     for i, midi_note in enumerate(midi_notes):
#         # Update progress
#         progress_bar.progress((i + 1) / total_notes)
        
#         # Convert MIDI note to frequency (A4 = 440 Hz)
#         frequency = 440 * (2 ** ((midi_note - 69) / 12))

#         # Time array for the note
#         t = np.linspace(0., duration_per_note, int(
#             sample_rate * duration_per_note), endpoint=False)

#         # 1. Generate Complex Waveform (Richer tone by adding harmonics)
#         note_audio_float = np.zeros_like(t)
#         # Add fundamental frequency
#         note_audio_float += np.sin(2. * np.pi * frequency * t) * 1.0
#         # Add 3rd harmonic
#         note_audio_float += np.sin(2. * np.pi * (3 * frequency) * t) * 0.33
#         # Add 5th harmonic
#         note_audio_float += np.sin(2. * np.pi * (5 * frequency) * t) * 0.2

#         # Normalize the waveform
#         note_audio_float = note_audio_float / np.max(np.abs(note_audio_float)) * amplitude_multiplier

#         # 2. Apply ADSR Envelope for more realistic decay
#         note_audio_float = apply_adsr_envelope(
#             note_audio_float,
#             duration_per_note,
#             sample_rate,
#             attack=0.01,
#             decay=0.1,
#             sustain=0.1,
#             release=0.2 
#         )

#         # Convert to 16-bit integer format
#         note_audio_int16 = (note_audio_float * max_val).astype(np.int16)

#         # Add a silent gap between notes
#         gap_samples = int(sample_rate * gap_duration)
#         silent_gap = np.zeros(gap_samples, dtype=np.int16)

#         audio_data = np.append(audio_data, note_audio_int16)
#         audio_data = np.append(audio_data, silent_gap)

#     # Ensure the final audio length is roughly 1 minute
#     final_duration = len(audio_data) / sample_rate
#     status_placeholder.text(f"Music generation complete! (Duration: ~{final_duration:.2f}s)")
#     progress_bar.empty()


#     # Create a temporary file to save the generated audio
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
#         output_filename = tmpfile.name
#         # Use scipy.io.wavfile to write the 16-bit PCM audio
#         wavfile.write(output_filename, sample_rate, audio_data)

#     return output_filename


# # --------------------- Streamlit App Interface ---------------------

# st.title("AI Music Remix and Generation 🎶 (Improved 1-Minute Demo)")
# st.write(
#     "Upload a song, select a mood or genre, and let AI create a remix or new music! "
#     "(Demo generates an *approximately* 1-minute, richer piano tune.)"
# )

# uploaded_file = st.file_uploader(
#     "Upload your song (MP3, WAV, etc.)", type=["mp3", "wav", "ogg"])

# mood_genre = st.selectbox(
#     "Select a mood or genre",
#     ["Happy", "Sad", "Energetic", "Relaxing",
#      "Pop", "Rock", "Electronic", "Hip Hop"]
# )

# # Placeholder for progress/status message
# status_placeholder = st.empty()
# output_placeholder = st.empty()

# # Initialize session state for generated audio path
# if 'generated_audio_path' not in st.session_state:
#     st.session_state['generated_audio_path'] = None


# if st.button("Remix/Generate Music"):
#     # Clear previous status and output
#     status_placeholder.empty()
#     output_placeholder.empty()

#     if uploaded_file is not None:
#         # File Validation
#         file_size_limit_mb = 20
#         if uploaded_file.size > file_size_limit_mb * 1024 * 1024:
#             status_placeholder.error(
#                 f"Uploaded file is too large. Please upload a file smaller than {file_size_limit_mb} MB.")
#         else:
#             status_placeholder.text("Uploading file...")
#             time.sleep(0.5)

#             uploaded_file_path = None
#             try:
#                 # 1. Save the uploaded file to a temporary location
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     uploaded_file_path = tmp_file.name

#                 # 2. Call the music generation function
#                 generated_audio_path = generate_music(
#                     uploaded_file_path, mood_genre, status_placeholder)

#                 # 3. Store the generated audio path in session state
#                 st.session_state['generated_audio_path'] = generated_audio_path
#                 status_placeholder.success("Music generation successful! Play the result below.")

#             except Exception as e:
#                 status_placeholder.error(
#                     f"An unexpected error occurred during music generation: {e}")
#                 st.session_state['generated_audio_path'] = None # Clear path on error

#             finally:
#                 # Clean up the temporary uploaded file
#                 if uploaded_file_path and os.path.exists(uploaded_file_path):
#                     os.remove(uploaded_file_path)

#     else:
#         # If no file is uploaded, still proceed with generating the demo tune
#         try:
#             # Pass placeholder values for the function arguments not used in demo
#             generated_audio_path = generate_music("No file uploaded", mood_genre, status_placeholder)
#             st.session_state['generated_audio_path'] = generated_audio_path
#             status_placeholder.success("Demo music generation successful! Play the result below.")
#         except Exception as e:
#             status_placeholder.error(f"Error during demo generation: {e}")
#             st.session_state['generated_audio_path'] = None


# # --------------------- Display Output ---------------------

# st.subheader("Output Music")

# # Display the audio player if generated audio is available
# audio_path = st.session_state['generated_audio_path']
# if audio_path and os.path.exists(audio_path):
#     try:
#         # Read the generated audio file into bytes
#         with open(audio_path, 'rb') as f:
#             audio_bytes = f.read()

#         output_placeholder.audio(audio_bytes, format='audio/wav')

#         # Add a download button
#         st.download_button(
#             label="Download Generated Music",
#             data=audio_bytes,
#             file_name="generated_music.wav",
#             mime="audio/wav"
#         )
#     except Exception as e:
#         output_placeholder.error(
#             f"An error occurred while loading the generated audio for playback: {e}")

# else:
#     output_placeholder.info("Click 'Remix/Generate Music' to see the output here.")

import streamlit as st
import os
import tempfile
import numpy as np
import time
import scipy.io.wavfile as wavfile
import math
# NOTE: The full analysis of arbitrary audio files (MP3/WAV) using libraries like
# Librosa is not supported in this environment. The following function is a
# SIMULATION of an audio analysis step.

def load_and_analyze_audio(uploaded_file_path):
    """
    SIMULATION: In a real AI music application, this function would use
    libraries like Librosa to analyze the uploaded track (pitch, tempo,
    timbre, etc.).

    For this demo, we return a fixed, simulated analysis result.
    If a real file is present, we return a simulated offset to "mimic" analysis.
    If no file is present, we return a default offset.
    """
    if uploaded_file_path != "No file uploaded" and os.path.exists(uploaded_file_path):
        # Simulate finding a lower key offset, common for a remix/cover version
        simulated_key_offset = -3 # Transpose down a minor third (e.g., C -> A)
        simulated_tempo_factor = 1.0 # Keep tempo same
        return simulated_key_offset, simulated_tempo_factor
    else:
        # Default offset if no file is uploaded
        return 0, 1.0

# --- ADSR Envelope Function (Unchanged) ---
def apply_adsr_envelope(audio_data, duration, sample_rate, attack=0.01, decay=0.1, sustain=0.1, release=0.1):
    """Applies a simple ADSR envelope to a single note's audio data."""
    total_samples = len(audio_data)

    # Calculate samples for each stage
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    attack_env = np.linspace(0.0, 1.0, attack_samples)
    decay_env = np.linspace(1.0, sustain, decay_samples)

    sustain_samples = total_samples - attack_samples - decay_samples
    if sustain_samples < 0:
        sustain_samples = 0

    sustain_env = np.full(sustain_samples, sustain)
    envelope = np.concatenate([attack_env, decay_env, sustain_env])[:total_samples]

    fade_samples = min(release_samples, total_samples)
    if fade_samples > 0:
        start_level = envelope[-fade_samples]
        envelope[-fade_samples:] *= np.linspace(start_level / sustain if sustain > 0 else 1.0, 0.0, fade_samples)

    return audio_data * envelope[:total_samples]

# Define the music generation/remixing logic function (Modified)
def generate_music(uploaded_file_path, mood_genre, status_placeholder):
    """
    Modified function that simulates using the uploaded file's analysis 
    to drive the generated music's pitch.
    """
    status_placeholder.text("Processing audio and analyzing input...")
    time.sleep(1)

    # --- SIMULATED INPUT ANALYSIS ---
    # The generated music parameters will be modified by the output of this function
    key_offset, tempo_factor = load_and_analyze_audio(uploaded_file_path)

    if uploaded_file_path != "No file uploaded":
        status_placeholder.text(
            f"Remixing/Generating {mood_genre} tune. Transposing by {key_offset} semitones.")
    else:
        status_placeholder.text(
            f"Generating default 1-minute {mood_genre} demo tune.")

    time.sleep(1)

    # --- Tone Generation Parameters ---
    sample_rate = 44100
    duration_per_note = 0.4 / tempo_factor # Adjust note duration based on simulated tempo factor
    gap_duration = 0.05 / tempo_factor

    total_note_duration = duration_per_note + gap_duration 

    # Base arpeggio pattern: C4, E4, G4, B4, G4, E4
    base_arpeggio_pattern = [60, 64, 67, 71, 67, 64] 
    
    # Apply the key offset from the simulated analysis
    arpeggio_pattern_transposed = [note + key_offset for note in base_arpeggio_pattern]
    
    # Calculate notes needed for 1 minute
    target_duration_seconds = 60
    notes_per_second = 1 / total_note_duration
    total_notes_needed = math.ceil(target_duration_seconds * notes_per_second)
    
    repetitions = math.ceil(total_notes_needed / len(arpeggio_pattern_transposed))
    midi_notes = (arpeggio_pattern_transposed * repetitions)[:total_notes_needed]

    audio_data = np.array([], dtype=np.int16)
    max_val = 32767
    amplitude_multiplier = 0.5 

    total_notes = len(midi_notes)
    
    progress_bar = st.progress(0)

    for i, midi_note in enumerate(midi_notes):
        progress_bar.progress((i + 1) / total_notes)
        
        frequency = 440 * (2 ** ((midi_note - 69) / 12))

        t = np.linspace(0., duration_per_note, int(
            sample_rate * duration_per_note), endpoint=False)

        # Generate Complex Waveform
        note_audio_float = np.zeros_like(t)
        note_audio_float += np.sin(2. * np.pi * frequency * t) * 1.0
        note_audio_float += np.sin(2. * np.pi * (3 * frequency) * t) * 0.33
        note_audio_float += np.sin(2. * np.pi * (5 * frequency) * t) * 0.2

        note_audio_float = note_audio_float / np.max(np.abs(note_audio_float)) * amplitude_multiplier

        # Apply ADSR Envelope
        note_audio_float = apply_adsr_envelope(
            note_audio_float,
            duration_per_note,
            sample_rate,
            attack=0.01,
            decay=0.1,
            sustain=0.1,
            release=0.2 
        )

        note_audio_int16 = (note_audio_float * max_val).astype(np.int16)

        gap_samples = int(sample_rate * gap_duration)
        silent_gap = np.zeros(gap_samples, dtype=np.int16)

        audio_data = np.append(audio_data, note_audio_int16)
        audio_data = np.append(audio_data, silent_gap)

    final_duration = len(audio_data) / sample_rate
    status_placeholder.text(f"Music generation complete! (Duration: ~{final_duration:.2f}s, Offset: {key_offset} semitones)")
    progress_bar.empty()


    # Create a temporary file to save the generated audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        output_filename = tmpfile.name
        wavfile.write(output_filename, sample_rate, audio_data)

    return output_filename


# --------------------- Streamlit App Interface (Unchanged) ---------------------

st.title("AI Music Remix and Generation 🎶 (Input-Simulated Demo)")
st.write(
    "Upload a song. The output tune's **pitch will be lowered** if a file is uploaded, "
    "simulating a remix that *mimics* the input's characteristics."
    "(Demo generates an *approximately* 1-minute, richer piano tune.)"
)

uploaded_file = st.file_uploader(
    "Upload your song (MP3, WAV, etc.)", type=["mp3", "wav", "ogg"])

mood_genre = st.selectbox(
    "Select a mood or genre",
    ["Happy", "Sad", "Energetic", "Relaxing",
     "Pop", "Rock", "Electronic", "Hip Hop"]
)

# Placeholder for progress/status message
status_placeholder = st.empty()
output_placeholder = st.empty()

# Initialize session state for generated audio path
if 'generated_audio_path' not in st.session_state:
    st.session_state['generated_audio_path'] = None


if st.button("Remix/Generate Music"):
    # Clear previous status and output
    status_placeholder.empty()
    output_placeholder.empty()

    if uploaded_file is not None:
        # File Validation
        file_size_limit_mb = 20
        if uploaded_file.size > file_size_limit_mb * 1024 * 1024:
            status_placeholder.error(
                f"Uploaded file is too large. Please upload a file smaller than {file_size_limit_mb} MB.")
        else:
            status_placeholder.text("Uploading file...")
            time.sleep(0.5)

            uploaded_file_path = None
            try:
                # 1. Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    uploaded_file_path = tmp_file.name

                # 2. Call the music generation function
                generated_audio_path = generate_music(
                    uploaded_file_path, mood_genre, status_placeholder)

                # 3. Store the generated audio path in session state
                st.session_state['generated_audio_path'] = generated_audio_path
                status_placeholder.success("Music generation successful! Play the result below.")

            except Exception as e:
                status_placeholder.error(
                    f"An unexpected error occurred during music generation: {e}")
                st.session_state['generated_audio_path'] = None # Clear path on error

            finally:
                # Clean up the temporary uploaded file
                if uploaded_file_path and os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path)

    else:
        # If no file is uploaded, still proceed with generating the demo tune
        try:
            # Pass placeholder values for the function arguments not used in demo
            generated_audio_path = generate_music("No file uploaded", mood_genre, status_placeholder)
            st.session_state['generated_audio_path'] = generated_audio_path
            status_placeholder.success("Demo music generation successful! Play the result below.")
        except Exception as e:
            status_placeholder.error(f"Error during demo generation: {e}")
            st.session_state['generated_audio_path'] = None


# --------------------- Display Output ---------------------

st.subheader("Output Music")

# Display the audio player if generated audio is available
audio_path = st.session_state['generated_audio_path']
if audio_path and os.path.exists(audio_path):
    try:
        # Read the generated audio file into bytes
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        output_placeholder.audio(audio_bytes, format='audio/wav')

        # Add a download button
        st.download_button(
            label="Download Generated Music",
            data=audio_bytes,
            file_name="generated_music_remix.wav",
            mime="audio/wav"
        )
    except Exception as e:
        output_placeholder.error(
            f"An error occurred while loading the generated audio for playback: {e}")

else:
    output_placeholder.info("Click 'Remix/Generate Music' to see the output here.")
