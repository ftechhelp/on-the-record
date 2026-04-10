# on-the-record

A cross-platform CLI tool that captures all audio playing through your system's output device (speakers/headphones), transcribes it with speaker diarization using OpenAI, and writes timestamped output to a file.

## Features

- Captures system audio via loopback (what you hear through your speakers)
- Speaker diarization — identifies who is speaking
- Multiple output formats: plain text, Markdown, JSON
- Configurable chunk size for real-time transcription
- Silence detection to skip quiet periods and save API costs
- Graceful start/stop via Ctrl+C

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **OpenAI API key** with billing enabled at [platform.openai.com](https://platform.openai.com/api-keys)
  - Note: A ChatGPT Plus subscription does **not** include API credits. You need a separate API key.

### Platform-specific audio setup

| OS | Setup Required |
|----|---------------|
| **Windows** | None — WASAPI loopback works natively |
| **Linux** | None — PulseAudio/PipeWire monitor sources work natively |
| **macOS** | Install [BlackHole](https://existential.audio/blackhole/) (see below) |

#### macOS setup (BlackHole)

1. Install BlackHole: `brew install blackhole-2ch` or download from [existential.audio/blackhole](https://existential.audio/blackhole/)
2. Open **Audio MIDI Setup** (search in Spotlight)
3. Click **+** → **Create Multi-Output Device**
4. Check both your speakers/headphones and **BlackHole 2ch**
5. Set the Multi-Output Device as your system output
6. When running on-the-record, select the BlackHole device: `--device BlackHole`

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/on-the-record.git
cd on-the-record

# Install with uv
uv sync
```

After `uv sync`, you have three ways to run the tool:

```bash
# Option 1: Use uv run (recommended — no activation needed)
uv run on-the-record start

# Option 2: Activate the virtual environment first
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
on-the-record start

# Option 3: Install globally so it works from anywhere
uv tool install .
on-the-record start
```

<details>
<summary>Alternative: install with pip</summary>

```bash
pip install .
on-the-record start
```
</details>

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='sk-...'
```

## Usage

### Start transcribing

```bash
# Basic usage — transcribes to ./transcript_<timestamp>.txt
uv run on-the-record start

# Specify output file and format
uv run on-the-record start --output ~/notes/meeting.md --format md

# Use JSON output
uv run on-the-record start --output ./call.json --format json

# Custom chunk size (30 seconds)
uv run on-the-record start --chunk-size 30

# Use a specific audio device
uv run on-the-record start --device "BlackHole"

# Disable speaker diarization (faster, cheaper)
uv run on-the-record start --no-diarize

# Combine options
uv run on-the-record start -o ./meeting.md -f md -c 20 -d "BlackHole"
```

> If you activated the venv or installed globally, you can drop the `uv run` prefix.

Press **Ctrl+C** to stop recording. The final chunk will be transcribed before exit.

### List audio devices

```bash
uv run on-the-record list-devices
```

Shows all available input and loopback devices so you can pick the right `--device` value.

### Using `python -m`

```bash
python -m on_the_record start --output notes.txt
python -m on_the_record list-devices
```

## CLI Reference

```
on-the-record start [OPTIONS]

Options:
  -o, --output PATH       Output file path (default: ./transcript_<timestamp>.<format>)
  -f, --format FORMAT     Output format: txt, md, json (default: txt)
  -c, --chunk-size SECS   Audio chunk length in seconds (default: 15)
  -d, --device NAME       Audio device name (substring match)
  -m, --model MODEL       OpenAI model (default: gpt-4o-transcribe)
  --diarize               Enable speaker diarization (default)
  --no-diarize            Disable speaker diarization
  --version               Show version
```

## Output formats

### Plain text (`.txt`)
```
[00:00:00] Speaker 1: Hello, welcome to the meeting.
[00:00:03] Speaker 2: Thanks for having me.
```

### Markdown (`.md`)
```markdown
# Transcript

**[00:00:00]** **Speaker 1**: Hello, welcome to the meeting.

**[00:00:03]** **Speaker 2**: Thanks for having me.
```

### JSON (`.json`)
```json
[
  {
    "speaker": "Speaker 1",
    "text": "Hello, welcome to the meeting.",
    "start": 0.0,
    "end": 2.85,
    "start_formatted": "00:00:00",
    "end_formatted": "00:00:02"
  }
]
```

## Cost

The `gpt-4o-transcribe-diarize` model costs approximately **$0.006 per minute** of audio. A 1-hour meeting costs roughly $0.36.

Silence detection skips empty chunks to minimize API usage.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run directly
uv run on-the-record start
```

## Architecture

```
soundcard (loopback capture, 16kHz PCM)
  -> buffer into N-second chunks
  -> silence detection (skip quiet chunks)
  -> encode to WAV in-memory
  -> OpenAI gpt-4o-transcribe-diarize API
  -> parse diarized segments
  -> append to output file (txt/md/json)
```

## License

MIT
