"""Twin Mind — Analyze Module

Uses Gemma 4 on Cactus to transcribe audio and analyze video frames,
producing a structured understanding of what was seen and heard.
"""

import json

from cactus import cactus_complete, cactus_transcribe

FRAME_INTERVAL_SEC = 2


def transcribe_audio(model, audio_path):
    """Transcribe an audio file using Gemma 4's speech-to-text."""
    if not audio_path:
        return ""
    print("Transcribing audio...")
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    raw = cactus_transcribe(model, audio_path, prompt, None, None, None)
    transcript = json.loads(raw).get("response", "")
    print(f"\nTranscript: {transcript}\n")
    return transcript


def analyze_frames(model, frame_paths, transcript=""):
    """Analyze each frame for visual context, building on prior observations."""
    transcript_ctx = f'Audio transcript from this session: "{transcript}"\n\n' if transcript else ""
    n = len(frame_paths)
    observations = []

    for i, frame_path in enumerate(frame_paths):
        print(f"Analyzing frame {i + 1}/{n}...")

        prior = ""
        if observations:
            prior = "Previous observations:\n" + "\n".join(
                f"  Frame {o['frame']}: {o['observation']}" for o in observations
            ) + "\n\n"

        content = (
            f"{transcript_ctx}"
            f"{prior}"
            f"This is frame {i + 1} of {n}, captured at {i * FRAME_INTERVAL_SEC}s.\n\n"
            f"Carefully scan the entire frame, including edges and background.\n"
            f"Describe concisely:\n"
            f"- ALL people visible, even if partially cropped — for each person: "
            f"approximate position (left/center/right), facial expression, body language, emotional tone\n"
            f"- How many people are in the frame total\n"
            f"- Objects, text, and environment\n"
            f"- Changes from the previous frame"
        )

        messages = [{"role": "user", "content": content, "images": [frame_path]}]
        response = cactus_complete(
            model,
            json.dumps(messages),
            json.dumps({"max_tokens": 350}),
            None, None,
        )
        result = json.loads(response)["response"]
        observations.append({"frame": i + 1, "observation": result})
        print(f"  {result}\n")

    return observations


def summarize_session(transcript, observations):
    """Combine transcript and visual observations into a session summary dict."""
    return {
        "transcript": transcript,
        "observations": observations,
        "visual_summary": " | ".join(o["observation"] for o in observations),
    }

