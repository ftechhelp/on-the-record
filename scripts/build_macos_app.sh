#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_SOURCE_DIR="$ROOT_DIR/macos/OnTheRecordMenuBar"
APP_NAME="On The Record.app"
DIST_DIR="$ROOT_DIR/dist"
APP_DIR="$DIST_DIR/$APP_NAME"
STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/on-the-record-app.XXXXXX")"
STAGING_APP_DIR="$STAGING_DIR/$APP_NAME"
CONTENTS_DIR="$STAGING_APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

UV_PATH="${UV_PATH:-$(command -v uv || true)}"
if [[ -z "$UV_PATH" ]]; then
  echo "uv was not found. Install uv or run with UV_PATH=/path/to/uv." >&2
  exit 1
fi

echo "Building Swift menu bar app..."
swift build -c release --package-path "$APP_SOURCE_DIR"
BIN_DIR="$(swift build -c release --package-path "$APP_SOURCE_DIR" --show-bin-path)"

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cp "$BIN_DIR/OnTheRecordMenuBar" "$MACOS_DIR/OnTheRecordMenuBar"
cp "$APP_SOURCE_DIR/Info.plist" "$CONTENTS_DIR/Info.plist"

cat > "$RESOURCES_DIR/EngineConfig.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>RepoRoot</key>
  <string>$ROOT_DIR</string>
  <key>UVPath</key>
  <string>$UV_PATH</string>
</dict>
</plist>
PLIST

chmod +x "$MACOS_DIR/OnTheRecordMenuBar"
xattr -cr "$STAGING_APP_DIR"
codesign --force --deep --sign - "$STAGING_APP_DIR" >/dev/null
xattr -cr "$STAGING_APP_DIR"

rm -rf "$APP_DIR"
mkdir -p "$DIST_DIR"
ditto --norsrc "$STAGING_APP_DIR" "$APP_DIR"
xattr -cr "$APP_DIR"
xattr -d com.apple.FinderInfo "$APP_DIR" 2>/dev/null || true
xattr -d 'com.apple.fileprovider.fpfs#P' "$APP_DIR" 2>/dev/null || true

echo "Built $APP_DIR"
echo "Open it with: open '$APP_DIR'"
