import streamlit as st
import os
import tempfile
import numpy as np
import time
import scipy.io.wavfile as wavfile
import math

# --- ADSR Envelope Function ---
def apply_adsr_envelope(audio_data, duration, sample_rate, attack=0.1, decay=0.1, sustain=0.5, release=0.3):
    """Applies a simple ADSR envelope to a single note's audio data."""
    total_samples = len(audio_data)

    # Calculate samples for each stage
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)

    # The current code generates a full-duration tone, so we assume no explicit 'sustain time' but a 'sustain level'
    # We will treat the initial duration as the total of A, D, and the effective playing time before Release.

    # Re-normalize times to the note's duration for a simple implementation
    # This assumes the tone itself is only as long as the note's duration.
    # We will use A, D, and S level applied across the note's duration, with a fade-out (Release) at the end.

    attack_env = np.linspace(0.0, 1.0, attack_samples)
    decay_env = np.linspace(1.0, sustain, decay_samples)

    # Calculate sustain samples based on remaining time
    sustain_samples = total_samples - attack_samples - decay_samples
    if sustain_samples < 0:
         sustain_samples = 0

    sustain_env = np.full(sustain_samples, sustain)

    # Combine envelope stages. We omit a separate release stage for this simple note generation,
    # and just ensure a smooth decay to zero at the very end.
    envelope = np.concatenate([attack_env, decay_env, sustain_env])[:total_samples]

    # Apply a final smooth fade to zero for the last 10% of the note's duration (like a release)
    fade_samples = min(int(0.1 * total_samples), total_samples)
    if fade_samples > 0:
        envelope[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

    return audio_data * envelope[:total_samples]

# Define the music generation/remixing logic function
def generate_music(uploaded_file_path, mood_genre, status_placeholder):
    """
    Placeholder function for AI music generation/remixing with progress updates,
    generating an improved piano-like sequence for demonstration.
    """
    status_placeholder.text("Processing audio...")
    time.sleep(1)

    status_placeholder.text(
        f"Generating an improved {mood_genre} piano tune...")
    time.sleep(1)

    # --- Improved Piano Tone Generation for Demonstration ---
    sample_rate = 44100
    duration_per_note = 0.6  # seconds per note
    midi_notes = [60, 64, 67, 72, 67, 64, 60]  # MIDI notes: C4, E4, G4, C5, G4, E4, C4

    audio_data = np.array([], dtype=np.int16)
    max_val = 32767
    amplitude_multiplier = 0.5 # Overall volume

    for midi_note in midi_notes:
        # Convert MIDI note to frequency (A4 = 440 Hz)
        frequency = 440 * (2 ** ((midi_note - 69) / 12))

        # Time array for the note
        t = np.linspace(0., duration_per_note, int(
            sample_rate * duration_per_note), endpoint=False)

        # 1. Generate Complex Waveform (Sawtooth/Square-like by adding harmonics)
        # Adding odd harmonics gives a square-wave quality, which is richer than a sine wave.
        note_audio_float = np.zeros_like(t)

        # Add fundamental frequency
        note_audio_float += np.sin(2. * np.pi * frequency * t) * 1.0
        # Add 3rd harmonic (less volume)
        note_audio_float += np.sin(2. * np.pi * (3 * frequency) * t) * 0.33
        # Add 5th harmonic (even less volume)
        note_audio_float += np.sin(2. * np.pi * (5 * frequency) * t) * 0.2

        # Normalize the waveform before applying the envelope
        note_audio_float = note_audio_float / np.max(np.abs(note_audio_float)) * amplitude_multiplier

        # 2. Apply ADSR Envelope for more realistic decay
        # Adjusted A, D, S for a faster, piano-like pluck and decay.
        note_audio_float = apply_adsr_envelope(
            note_audio_float,
            duration_per_note,
            sample_rate,
            attack=0.01,  # Quick attack
            decay=0.1,    # Fast decay
            sustain=0.1,  # Low sustain level
            release=0.1   # Quick release
        )

        # Convert to 16-bit integer format
        note_audio_int16 = (note_audio_float * max_val).astype(np.int16)

        # Add a small silent gap between notes
        gap_duration = 0.1 # seconds
        gap_samples = int(sample_rate * gap_duration)
        silent_gap = np.zeros(gap_samples, dtype=np.int16)

        audio_data = np.append(audio_data, note_audio_int16)
        audio_data = np.append(audio_data, silent_gap)


    # Create a temporary file to save the generated audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        output_filename = tmpfile.name
        # Use scipy.io.wavfile to write the 16-bit PCM audio
        wavfile.write(output_filename, sample_rate, audio_data)

    status_placeholder.text("Music generation complete!")
    return output_filename


# --------------------- Streamlit App Interface ---------------------

st.title("AI Music Remix and Generation 🎶 (Improved Sound)")
st.write("Upload a song, select a mood or genre, and let AI create a remix or new music! (Demo generates a more realistic piano tune)")

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
            file_name="generated_music.wav",
            mime="audio/wav"
        )
    except Exception as e:
        output_placeholder.error(
            f"An error occurred while loading the generated audio for playback: {e}")

else:
    output_placeholder.info("Click 'Remix/Generate Music' to see the output here.")

