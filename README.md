# on-the-record

A cross-platform CLI tool that captures all audio playing through your system's output device (speakers/headphones), transcribes it with speaker diarization using OpenAI, and writes timestamped output to a file.

## Features

- Captures system audio via loopback (what you hear through your speakers) plus your microphone
- Buffered capture pipeline keeps recording even if transcription falls behind
- **Native macOS support** — uses ScreenCaptureKit on macOS 13+, no virtual audio device needed
- Speaker diarization — identifies who is speaking
- Multiple output formats: plain text, Markdown, JSON
- Optional Gemini-generated Markdown study documents after recording stops
- Optional Obsidian export for Gemini study documents into a configured vault folder
- Configurable chunk size for real-time transcription
- Silence detection to skip quiet periods and save API costs
- Graceful start/stop via Ctrl+C

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **OpenAI API key** with billing enabled at [platform.openai.com](https://platform.openai.com/api-keys)
  - Note: A ChatGPT Plus subscription does **not** include API credits. You need a separate API key.
- **Gemini API key** if you want automatic Markdown study documents after recording

### Platform-specific audio setup

| OS | Setup Required |
|----|---------------|
| **Windows** | None — WASAPI loopback works natively |
| **Linux** | None — PulseAudio/PipeWire monitor sources work natively |
| **macOS 13+** | None — ScreenCaptureKit captures system audio natively (grant Screen Recording permission when prompted) |
| **macOS 12 and older** | Install [BlackHole](https://existential.audio/blackhole/) (see below) |

#### macOS 13+ (Ventura and later)

On macOS 13+, on-the-record uses Apple's **ScreenCaptureKit** framework to capture system audio directly. No virtual audio device is needed.

- The first time you run `on-the-record start`, macOS will prompt you to grant **Screen Recording** permission (this is how Apple gates system audio access — even though we only capture audio).
- Go to **System Settings > Privacy & Security > Screen Recording** and enable access for your terminal app (Terminal, iTerm2, etc.).
- You will still hear audio through your speakers normally — nothing changes about your audio setup.

```bash
# Just works — no device selection needed
uv run on-the-record start

# Test that audio is flowing
uv run on-the-record test-audio
```

<details>
<summary>macOS 12 and older (BlackHole fallback)</summary>

#### macOS setup (BlackHole)

BlackHole is a virtual audio driver that creates a "pipe" between apps. By itself it doesn't capture speaker output — you need a **Multi-Output Device** to split your audio to both your speakers and BlackHole simultaneously.

1. Install BlackHole: `brew install blackhole-2ch` or download from [existential.audio/blackhole](https://existential.audio/blackhole/)
2. Open **Audio MIDI Setup** (Cmd+Space → search "Audio MIDI Setup")
3. **Match sample rates before creating the Multi-Output Device** — this is critical:
   - Click **BlackHole 2ch** in the left sidebar → set format to **44100 Hz** (bottom right)
   - Click your **speakers/headphones** → set them to the same **44100 Hz**
   - If the sample rates don't match, the Multi-Output Device will be **greyed out** and unusable
4. Click **+** at the bottom left → **Create Multi-Output Device**
5. In the right panel, check **both** of these — **order matters**:
   - Check your **speakers/headphones first** (e.g. "MacBook Air Speakers") — this makes them the clock source
   - Then check **BlackHole 2ch**
   - Enable **Drift Correction** for BlackHole 2ch (the second device)
   - The first device checked becomes the clock master and primary audio output — if you don't hear sound, this is probably why
6. Set the Multi-Output Device as your system output. Pick **one** of these methods:
   - **Audio MIDI Setup:** Right-click the Multi-Output Device → **"Use This Device For Sound Output"**
   - **Menu bar:** Hold **Option** and click the volume/sound icon → select Multi-Output Device
   - **Terminal:** `SwitchAudioSource -s "Multi-Output Device"` (install first: `brew install switchaudio-osx`)
   - **System Settings:** System Settings → Sound → Output → Multi-Output Device (this can sometimes spin forever — use one of the methods above instead)
7. Run on-the-record: `uv run on-the-record start --device "BlackHole 2ch"`

> **Note:** The Multi-Output Device **disables the system volume keys** (the menu bar slider disappears).
> To control volume, open Audio MIDI Setup → click Multi-Output Device → adjust the volume slider
> next to your speakers. When you're done recording, switch your output back to your normal speakers.

#### macOS troubleshooting (BlackHole)

| Problem | Cause | Fix |
|---------|-------|-----|
| Multi-Output Device is greyed out | Sample rate mismatch between BlackHole and your speakers | Set both to the same sample rate (e.g. 44100 Hz) in Audio MIDI Setup, then delete and recreate the Multi-Output Device |
| No audio from speakers with Multi-Output | Speakers aren't the clock source, or volume is at zero | In Audio MIDI Setup, delete the Multi-Output Device and recreate it — check your **speakers first**, then BlackHole. Also check the volume slider next to your speakers in the Multi-Output Device config |
| System volume keys don't work | Expected behavior with Multi-Output Devices | Adjust volume in Audio MIDI Setup → click Multi-Output Device → use the slider next to your speakers |
| System Settings spins when selecting Multi-Output | Known macOS bug | Use Audio MIDI Setup right-click, Option+click menu bar, or `SwitchAudioSource` in terminal instead |
| `on-the-record` says all chunks are silent | System output is not set to the Multi-Output Device | Verify with `SwitchAudioSource -c` or run `uv run on-the-record test-audio --device "BlackHole 2ch"` to diagnose |
| Audio is captured but very quiet | Volume level issue | Audio capture is digital and volume-independent; make sure the app playing audio isn't muted |
| `no audio device matching 'BlackHole' found` | BlackHole not installed or name mismatch | Run `uv run on-the-record list-devices` to see exact device names |

</details>

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

## Build a Windows executable

Build the executable on Windows. PyInstaller does not support cross-compiling a working Windows binary from macOS or Linux.

```bash
# Install build dependencies
uv sync --group build

# Produce dist/on-the-record.exe
uv run python scripts/build_windows_exe.py
```

The generated executable is a console app at `dist/on-the-record.exe` and uses the same `OPENAI_API_KEY` environment variable as the Python version.

If a project-root `.env` file exists when you build, it is bundled into `dist/on-the-record.exe`. Rebuild the exe after changing `.env` values that you want embedded.

## Configuration

Set your OpenAI API key as an environment variable:

```bash
$env:OPENAI_API_KEY = "sk-..."   # PowerShell
set OPENAI_API_KEY=sk-...         # cmd.exe

# macOS/Linux
export OPENAI_API_KEY='sk-...'
```

You can also put runtime keys in a project-root `.env` file:

```dotenv
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

Real environment variables take precedence over `.env` values. When running the exe, on-the-record also checks for a sidecar `.env` next to the exe and then for a `.env` bundled at build time. Keep `.env` private; embedding it in the exe also embeds those secrets in the binary.

To generate a Markdown study document after each recording, also set a Gemini API key:

```bash
$env:GEMINI_API_KEY = "..."   # PowerShell
set GEMINI_API_KEY=...         # cmd.exe

# macOS/Linux
export GEMINI_API_KEY='...'
```

If `GEMINI_API_KEY` is not set, recording and transcription still work; the study document step is skipped.

### Obsidian study-note export

on-the-record can copy each Gemini-generated study document into a configured Obsidian vault. The primary integration is a normal filesystem write into your vault folder, so Obsidian sync plugins and Obsidian itself can pick up the note naturally.

Configure your vault once:

```bash
uv run on-the-record config obsidian --vault ~/Documents/ObsidianVault --folder "OTR Study Notes"
```

Show or clear the saved Obsidian settings:

```bash
uv run on-the-record config obsidian --show
uv run on-the-record config obsidian --clear
```

You can also configure an optional external CLI hook that runs after export. Use `{file}` for the exported Markdown path and `{vault}` for the vault path. If no placeholder is used, the exported file path is appended to the command.

```bash
uv run on-the-record config obsidian --cli-command "obsidian open {file}"
uv run on-the-record config obsidian --clear-cli-command
```

When Obsidian is configured, study documents are exported after they are generated. Use `--no-obsidian` to skip export for one run, or use `--obsidian-vault` / `--obsidian-folder` to override the saved destination for one run.

## Usage

### Start transcribing

```bash
# Basic usage — records system audio plus your microphone and transcribes to ./transcript_<timestamp>.txt
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
If `GEMINI_API_KEY` is set, on-the-record then asks Gemini to turn the transcript into a Markdown study document. When you do not pass `--study-output`, Gemini also chooses a short searchable title for the filename. The default path is date-prefixed, for example `2026-05-01-neural-network-basics.md`.

```bash
# Write the Gemini study document to a custom path
uv run on-the-record start --study-output ./study-notes.md

# Disable the Gemini study document for one run
uv run on-the-record start --no-study-doc

# Use a different Gemini model
uv run on-the-record start --gemini-model gemini-2.5-pro

# Export the generated study document to your configured Obsidian vault
uv run on-the-record start --obsidian

# Skip Obsidian export for one run
uv run on-the-record start --no-obsidian

# Override the Obsidian destination for one run
uv run on-the-record start --obsidian-vault ~/Documents/ObsidianVault --obsidian-folder "Classes/OTR"
```

If `--study-output` is provided, on-the-record writes that exact local study-document path. If Obsidian export is enabled, that file is copied into the configured vault folder using the same basename.

### List audio devices

```bash
uv run on-the-record list-devices
```

Shows all available input and loopback devices so you can pick the right `--device` value.

### Test audio (diagnose issues)

```bash
# Check if audio is actually flowing through BlackHole (no API key needed)
uv run on-the-record test-audio --device "BlackHole 2ch"

# Record for 10 seconds instead of the default 5
uv run on-the-record test-audio --device "BlackHole 2ch" --seconds 10
```

Reports peak level, RMS level, and whether the silence threshold would be triggered. Use this to verify your Multi-Output Device setup is working before spending API credits.

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
  --study-doc             Generate a Gemini Markdown study document after recording (default when GEMINI_API_KEY is set)
  --no-study-doc          Disable Gemini study document generation
  --study-output PATH     Study document output path (default: Gemini-titled YYYY-MM-DD-<title>.md)
  --gemini-model MODEL    Gemini model for study document generation (default: gemini-2.5-flash)
  --obsidian              Export the Gemini study document to the configured Obsidian vault
  --no-obsidian           Disable Obsidian export for one run
  --obsidian-vault PATH   Override the saved Obsidian vault path for one run
  --obsidian-folder PATH  Override the saved vault-relative study folder for one run
  --obsidian-cli-command CMD
                          Override the optional post-export CLI hook for one run
  --version               Show version
```

```
on-the-record config obsidian [OPTIONS]

Options:
  --vault PATH            Path to your Obsidian vault
  --folder PATH           Vault-relative folder for OTR study documents
  --cli-command CMD       Optional post-export command. Supports {file} and {vault}
  --clear-cli-command     Remove the configured CLI hook
  --show                  Show the current Obsidian config
  --clear                 Clear the Obsidian config
```

```
on-the-record test-audio [OPTIONS]

Options:
  -d, --device NAME       Audio device name (substring match)
  -s, --seconds SECS      Seconds to record (default: 5)
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
Audio capture (platform-dependent):
  macOS 13+:  ScreenCaptureKit (native system audio, no setup)
  macOS <13:  soundcard + BlackHole virtual device
  Linux:      soundcard + PulseAudio/PipeWire loopback
  Windows:    soundcard + WASAPI loopback

  -> buffer into N-second chunks (16kHz PCM)
  -> silence detection (skip quiet chunks)
  -> encode to WAV in-memory
  -> OpenAI gpt-4o-transcribe-diarize API
  -> parse diarized segments
  -> append to output file (txt/md/json)
```

## License

MIT
