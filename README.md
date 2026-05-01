# on-the-record

Record system audio and your microphone, transcribe it with OpenAI, and write a timestamped transcript as `txt`, `md`, or `json`.

It is built as a cross-platform CLI for meetings, lectures, calls, videos, and any other audio playing through your computer. After transcription, it can optionally ask Gemini to turn the transcript into a Markdown study document and copy that note into an Obsidian vault.

## What it does

- Captures system audio plus your microphone.
- Streams audio in chunks so recording can continue while transcription runs.
- Uses native ScreenCaptureKit on macOS 13+.
- Uses WASAPI loopback on Windows and PulseAudio/PipeWire monitor sources on Linux.
- Supports speaker diarization through OpenAI transcription models.
- Skips silent chunks to reduce API cost.
- Writes `txt`, `md`, or `json` transcripts.
- Can generate Gemini Markdown study notes after recording.
- Can export generated study notes into Obsidian.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for transcription
- A Gemini API key only if you want study-document generation

Set keys in your shell:

```bash
# macOS/Linux
export OPENAI_API_KEY='sk-...'
export GEMINI_API_KEY='...'
```

```powershell
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
$env:GEMINI_API_KEY = "..."
```

You can also create a project-root `.env` file:

```dotenv
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

Real environment variables take precedence over `.env` values. `list-devices` and `test-audio` do not require `OPENAI_API_KEY`.

## Install

```bash
git clone https://github.com/your-username/on-the-record.git
cd on-the-record
uv sync
```

Run through `uv`:

```bash
uv run on-the-record start
```

Or activate the virtual environment first:

```bash
source .venv/bin/activate
on-the-record start
```

## Quick Start

```bash
# See available capture devices
uv run on-the-record list-devices

# Check audio levels without spending API credits
uv run on-the-record test-audio

# Start recording and transcribing
uv run on-the-record start
```

Press `Ctrl+C` to stop. The final buffered chunk is transcribed before the program exits.

By default, transcripts are written to `./transcript_<timestamp>.txt`. If `GEMINI_API_KEY` is set, a Markdown study document is generated after a transcript with at least one segment is written.

## Common Commands

```bash
# Markdown transcript
uv run on-the-record start --output ./meeting.md --format md

# JSON transcript
uv run on-the-record start --output ./meeting.json --format json

# Longer transcription chunks
uv run on-the-record start --chunk-size 30

# Use a specific device name or substring
uv run on-the-record start --device "BlackHole 2ch"

# Disable diarization
uv run on-the-record start --no-diarize

# Disable Gemini study-note generation for one run
uv run on-the-record start --no-study-doc

# Choose a study-note path
uv run on-the-record start --study-output ./study-notes.md
```

## Platform Setup

| OS | What to know |
| --- | --- |
| macOS 13+ | Uses ScreenCaptureKit. Grant Screen Recording permission to your terminal app when prompted. |
| macOS 12 or older | Install BlackHole and set a Multi-Output Device that includes your speakers and BlackHole. |
| Windows | Uses WASAPI loopback. No virtual audio device is normally needed. |
| Linux | Uses PulseAudio/PipeWire monitor sources. No virtual audio device is normally needed. |

For macOS 13+, run `uv run on-the-record test-audio` after granting permission. If the result is silent, play audio and try again.

For macOS 12 or older:

1. Install BlackHole, for example `brew install blackhole-2ch`.
2. Open Audio MIDI Setup.
3. Set BlackHole and your speakers/headphones to the same sample rate.
4. Create a Multi-Output Device.
5. Add your speakers/headphones first, then BlackHole.
6. Enable Drift Correction for BlackHole.
7. Set the Multi-Output Device as your system output.
8. Test with `uv run on-the-record test-audio --device "BlackHole 2ch"`.

Multi-Output Devices can disable the normal macOS volume keys. Adjust volume in Audio MIDI Setup or switch back to your normal output when finished.

## Study Documents

If `GEMINI_API_KEY` is set, `start` generates a Markdown study document after recording. Without `--study-output`, Gemini also chooses a short title and the file is named like `2026-05-01-topic-name.md`.

```bash
# Disable study documents
uv run on-the-record start --no-study-doc

# Use a specific Gemini model
uv run on-the-record start --gemini-model gemini-3-flash-preview

# Write the study document to a specific path
uv run on-the-record start --study-output ./notes/study.md
```

The default Gemini model is `gemini-3-flash-preview`.

## Obsidian Export

Configure a vault once:

```bash
uv run on-the-record config obsidian --vault ~/Documents/ObsidianVault --folder "OTR Study Notes"
```

Then export generated study documents:

```bash
uv run on-the-record start --obsidian
```

Useful config commands:

```bash
uv run on-the-record config obsidian --show
uv run on-the-record config obsidian --clear
uv run on-the-record config obsidian --cli-command "obsidian open {file}"
uv run on-the-record config obsidian --clear-cli-command
```

Per-run overrides:

```bash
uv run on-the-record start --obsidian-vault ~/Documents/ObsidianVault --obsidian-folder "Classes/OTR"
uv run on-the-record start --no-obsidian
```

Obsidian export copies the generated Markdown file into the configured vault folder. The optional CLI command runs after export. Use `{file}` for the exported note path and `{vault}` for the vault path.

## CLI Reference

```text
on-the-record start [OPTIONS]

Options:
  -o, --output PATH            Transcript path. Defaults to ./transcript_<timestamp>.<format>
  -f, --format {txt,md,json}   Transcript format. Default: txt
  -c, --chunk-size SECONDS     Audio chunk length. Default: 15
  -d, --device NAME            Audio device name substring
  -m, --model MODEL            OpenAI model. Default: gpt-4o-transcribe
  --diarize                    Enable speaker diarization. Default
  --no-diarize                 Disable speaker diarization
  --study-doc                  Generate a Gemini study document. Default when GEMINI_API_KEY is set
  --no-study-doc               Disable Gemini study documents
  --study-output PATH          Study document path. Defaults to a Gemini-titled Markdown file
  --gemini-model MODEL         Gemini model. Default: gemini-3-flash-preview
  --obsidian                   Export the study document to the configured Obsidian vault
  --no-obsidian                Disable Obsidian export for this run
  --obsidian-vault PATH        Override the saved vault path for this run
  --obsidian-folder PATH       Override the saved vault-relative folder for this run
  --obsidian-cli-command CMD   Override the post-export CLI hook for this run
```

```text
on-the-record list-devices
```

```text
on-the-record test-audio [OPTIONS]

Options:
  -d, --device NAME            Audio device name substring
  -s, --seconds SECONDS        Seconds to record. Default: 5
```

```text
on-the-record config obsidian [OPTIONS]

Options:
  --vault PATH                 Path to your Obsidian vault
  --folder PATH                Vault-relative study-note folder
  --cli-command CMD            Optional post-export command with {file} and {vault}
  --clear-cli-command          Remove the configured CLI hook
  --show                       Show the current Obsidian config
  --clear                      Clear the Obsidian config
```

## Output Formats

Plain text:

```text
[00:00:00] Speaker 1: Hello, welcome to the meeting.
[00:00:03] Speaker 2: Thanks for having me.
```

Markdown:

```markdown
# Transcript

**[00:00:00]** **Speaker 1**: Hello, welcome to the meeting.

**[00:00:03]** **Speaker 2**: Thanks for having me.
```

JSON:

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

## Windows Executable

Build on Windows. PyInstaller does not cross-compile a working Windows executable from macOS or Linux.

```bash
uv sync --group build
uv run python scripts/build_windows_exe.py
```

The output is `dist/on-the-record.exe`.

If a root `.env` exists during the build, it is bundled into the executable. Rebuild after changing embedded values, and treat the executable as containing those secrets.

## Development

```bash
uv sync --group dev
uv run pytest
```

No lint or type-check command is configured.

## Architecture

```text
platform audio capture
  -> N-second 16 kHz chunks
  -> silence detection
  -> in-memory WAV encoding
  -> OpenAI transcription
  -> transcript writer
  -> optional Gemini study document
  -> optional Obsidian export
```

Main modules:

- `cli.py`: command parsing and recording workflow
- `audio.py`: shared recorder and chunk behavior
- `macos_audio.py`: ScreenCaptureKit backend
- `transcribe.py`: OpenAI transcription integration
- `study.py`: Gemini study-document generation
- `obsidian.py`: Obsidian export config and copy logic
- `writer.py`: `txt`, `md`, and `json` transcript writers

## License

MIT
