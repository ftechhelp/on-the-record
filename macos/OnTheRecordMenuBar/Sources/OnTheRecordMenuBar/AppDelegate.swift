import AppKit

struct EngineLaunchConfig {
    let executablePath: String
    let arguments: [String]
    let workingDirectory: URL
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private let engine = EngineProcess()
    private var statusItem: NSStatusItem?
    private var startMenuItem: NSMenuItem?
    private var stopMenuItem: NSMenuItem?
    private var revealMenuItem: NSMenuItem?
    private var windowController: SettingsWindowController?
    private var settings: AppSettings = AppDelegate.defaultSettings()
    private var isRecording = false
    private var lastTranscriptPath: String?

    func applicationDidFinishLaunching(_ notification: Notification) {
        settings = loadSettings()
        configureMenu()
        configureWindow()
        configureEngineCallbacks()
        startEngine()
    }

    func applicationWillTerminate(_ notification: Notification) {
        engine.stop()
    }

    private func configureMenu() {
        let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "OTR"
        statusItem.button?.image = NSImage(systemSymbolName: "waveform.circle", accessibilityDescription: "On The Record")
        statusItem.button?.imagePosition = .imageLeading

        let menu = NSMenu()

        let showItem = NSMenuItem(title: "Show On The Record", action: #selector(showWindow), keyEquivalent: "")
        showItem.target = self
        menu.addItem(showItem)

        let startItem = NSMenuItem(title: "Start Recording", action: #selector(startRecording), keyEquivalent: "")
        startItem.target = self
        menu.addItem(startItem)
        startMenuItem = startItem

        let stopItem = NSMenuItem(title: "Stop Recording", action: #selector(stopRecording), keyEquivalent: "")
        stopItem.target = self
        stopItem.isEnabled = false
        menu.addItem(stopItem)
        stopMenuItem = stopItem

        menu.addItem(.separator())

        let devicesItem = NSMenuItem(title: "List Devices", action: #selector(listDevices), keyEquivalent: "")
        devicesItem.target = self
        menu.addItem(devicesItem)

        let revealItem = NSMenuItem(title: "Reveal Last Transcript", action: #selector(revealLastTranscript), keyEquivalent: "")
        revealItem.target = self
        revealItem.isEnabled = false
        menu.addItem(revealItem)
        revealMenuItem = revealItem

        menu.addItem(.separator())

        let quitItem = NSMenuItem(title: "Quit", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
        self.statusItem = statusItem
    }

    private func configureWindow() {
        let controller = SettingsWindowController(settings: settings)
        controller.onSave = { [weak self] newSettings in
            self?.saveSettings(newSettings)
        }
        controller.onStart = { [weak self] in
            self?.startRecording()
        }
        controller.onStop = { [weak self] in
            self?.stopRecording()
        }
        controller.onChooseOutputFolder = { [weak self] in
            self?.chooseOutputFolder()
        }
        windowController = controller
    }

    private func configureEngineCallbacks() {
        engine.onEvent = { [weak self] event in
            DispatchQueue.main.async {
                self?.handleEngineEvent(event)
            }
        }
        engine.onLog = { [weak self] message in
            DispatchQueue.main.async {
                self?.appendLog(message)
            }
        }
        engine.onExit = { [weak self] status in
            DispatchQueue.main.async {
                self?.appendLog("Engine exited with status \(status)")
                self?.setRecording(false, status: "Engine stopped")
            }
        }
    }

    private func startEngine() {
        guard !engine.isRunning else { return }

        do {
            let config = loadEngineLaunchConfig()
            appendLog("Starting engine in \(config.workingDirectory.path)")
            try engine.start(
                executablePath: config.executablePath,
                arguments: config.arguments,
                workingDirectory: config.workingDirectory
            )
        } catch {
            appendLog("Could not start engine: \(error.localizedDescription)")
            showAlert("Could not start the recording engine", detail: error.localizedDescription)
        }
    }

    private func loadEngineLaunchConfig() -> EngineLaunchConfig {
        let environment = ProcessInfo.processInfo.environment
        var repoRoot = environment["ON_THE_RECORD_REPO_ROOT"]
        var uvPath = environment["ON_THE_RECORD_UV_PATH"]

        if let configURL = Bundle.main.url(forResource: "EngineConfig", withExtension: "plist"),
           let config = NSDictionary(contentsOf: configURL) as? [String: String] {
            repoRoot = repoRoot ?? config["RepoRoot"]
            uvPath = uvPath ?? config["UVPath"]
        }

        let workingDirectory = URL(fileURLWithPath: repoRoot ?? FileManager.default.currentDirectoryPath)
        if let uvPath, FileManager.default.isExecutableFile(atPath: uvPath) {
            return EngineLaunchConfig(
                executablePath: uvPath,
                arguments: ["run", "on-the-record-engine"],
                workingDirectory: workingDirectory
            )
        }

        return EngineLaunchConfig(
            executablePath: "/usr/bin/env",
            arguments: ["uv", "run", "on-the-record-engine"],
            workingDirectory: workingDirectory
        )
    }

    @objc private func showWindow() {
        windowController?.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func startRecording() {
        settings = windowController?.currentSettings ?? settings
        guard !settings.apiKey.isEmpty else {
            showWindow()
            showAlert("Add your OpenAI API key", detail: "Save an API key before starting a recording from the app.")
            return
        }

        saveSettings(settings, showConfirmation: false)
        startEngine()
        guard engine.isRunning else { return }

        let transcriptPath = nextTranscriptPath(settings: settings)
        lastTranscriptPath = transcriptPath
        windowController?.clearTranscript()
        setRecording(true, status: "Starting recording")

        let payload: JSONDictionary = [
            "api_key": settings.apiKey,
            "output_path": transcriptPath,
            "format": settings.format,
            "chunk_seconds": settings.chunkSeconds,
            "include_system_audio": settings.includesSystemAudio,
            "include_microphone": settings.includesMicrophone,
            "diarize": settings.diarize,
        ]

        do {
            try engine.send(command: "start_recording", payload: payload)
            appendLog("Recording to \(transcriptPath)")
        } catch {
            setRecording(false, status: "Could not start recording")
            appendLog("Start failed: \(error.localizedDescription)")
        }
    }

    @objc private func stopRecording() {
        do {
            try engine.send(command: "stop_recording")
            setRecording(false, status: "Stopping after current chunk")
        } catch {
            appendLog("Stop failed: \(error.localizedDescription)")
        }
    }

    @objc private func listDevices() {
        startEngine()
        do {
            try engine.send(command: "list_devices")
        } catch {
            appendLog("List devices failed: \(error.localizedDescription)")
        }
    }

    @objc private func revealLastTranscript() {
        guard let lastTranscriptPath else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: lastTranscriptPath)])
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }

    private func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: settings.outputDirectory)

        if panel.runModal() == .OK, let url = panel.url {
            settings.outputDirectory = url.path
            windowController?.apply(settings)
            saveSettings(settings)
        }
    }

    private func handleEngineEvent(_ event: JSONDictionary) {
        guard let eventName = event["event"] as? String else { return }

        switch eventName {
        case "ready":
            setRecording(false, status: "Ready")
            appendLog("Engine ready")
        case "start_accepted":
            setRecording(true, status: "Recording")
        case "recording_started":
            setRecording(true, status: "Recording")
        case "transcription_started":
            if let chunkIndex = event["chunk_index"] {
                appendLog("Transcribing chunk \(chunkIndex)")
            }
        case "segments_written":
            appendSegments(from: event)
        case "no_speech_detected":
            appendLog("No speech detected in chunk")
        case "recording_finished":
            setRecording(false, status: "Finished")
            revealMenuItem?.isEnabled = lastTranscriptPath != nil
            appendLog("Recording finished")
        case "recording_stopped":
            setRecording(false, status: "Stopped")
        case "devices":
            appendDevices(from: event)
        case "stop_requested":
            appendLog("Stop requested")
        case "stop_ignored":
            appendLog("No recording is running")
        case "recording_error", "fatal_error", "error":
            setRecording(false, status: "Error")
            appendLog(event["error"] as? String ?? "Unknown engine error")
        default:
            appendLog("Engine event: \(eventName)")
        }
    }

    private func appendSegments(from event: JSONDictionary) {
        guard let segments = event["segments"] as? [JSONDictionary] else { return }

        for segment in segments {
            let speaker = segment["speaker"] as? String ?? "Speaker"
            let text = segment["text"] as? String ?? ""
            let timestamp = formattedTime(segment["start"] as? Double)
            windowController?.appendTranscript("[\(timestamp)] \(speaker): \(text)")
        }
    }

    private func appendDevices(from event: JSONDictionary) {
        guard let devices = event["devices"] as? [JSONDictionary] else { return }
        if devices.isEmpty {
            appendLog("No devices returned")
            return
        }

        appendLog("Devices:")
        for device in devices {
            let name = device["name"] as? String ?? "Unknown"
            let isLoopback = (device["is_loopback"] as? Bool) == true ? "loopback" : "input"
            appendLog("- \(name) (\(isLoopback))")
        }
    }

    private func setRecording(_ recording: Bool, status: String) {
        isRecording = recording
        statusItem?.button?.title = recording ? "REC" : "OTR"
        startMenuItem?.isEnabled = !recording
        stopMenuItem?.isEnabled = recording
        windowController?.setStatus(status, isRecording: recording)
    }

    private func appendLog(_ message: String) {
        windowController?.appendLog(message)
    }

    private func saveSettings(_ newSettings: AppSettings, showConfirmation: Bool = true) {
        settings = newSettings
        UserDefaults.standard.set(newSettings.outputDirectory, forKey: "outputDirectory")
        UserDefaults.standard.set(newSettings.format, forKey: "format")
        UserDefaults.standard.set(newSettings.sourceMode, forKey: "sourceMode")
        UserDefaults.standard.set(newSettings.chunkSeconds, forKey: "chunkSeconds")
        UserDefaults.standard.set(newSettings.diarize, forKey: "diarize")

        do {
            try KeychainStore.save(newSettings.apiKey, account: "OPENAI_API_KEY")
            if showConfirmation {
                appendLog("Settings saved")
            }
        } catch {
            appendLog("Could not save API key to Keychain: \(error.localizedDescription)")
        }
    }

    private func loadSettings() -> AppSettings {
        let defaults = UserDefaults.standard
        var loadedSettings = AppDelegate.defaultSettings()
        loadedSettings.apiKey = KeychainStore.read(account: "OPENAI_API_KEY") ?? ""
        loadedSettings.outputDirectory = defaults.string(forKey: "outputDirectory") ?? loadedSettings.outputDirectory
        loadedSettings.format = defaults.string(forKey: "format") ?? loadedSettings.format
        loadedSettings.sourceMode = defaults.string(forKey: "sourceMode") ?? loadedSettings.sourceMode

        let savedChunkSeconds = defaults.integer(forKey: "chunkSeconds")
        if savedChunkSeconds > 0 {
            loadedSettings.chunkSeconds = savedChunkSeconds
        }
        if defaults.object(forKey: "diarize") != nil {
            loadedSettings.diarize = defaults.bool(forKey: "diarize")
        }

        return loadedSettings
    }

    private static func defaultSettings() -> AppSettings {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        let outputDirectory = documentsDirectory?
            .appendingPathComponent("On The Record", isDirectory: true)
            .path ?? FileManager.default.currentDirectoryPath

        return AppSettings(
            apiKey: "",
            outputDirectory: outputDirectory,
            format: "txt",
            sourceMode: "both",
            chunkSeconds: 15,
            diarize: true
        )
    }

    private func nextTranscriptPath(settings: AppSettings) -> String {
        let outputURL = URL(fileURLWithPath: settings.outputDirectory, isDirectory: true)
        try? FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let filename = "transcript_\(formatter.string(from: Date())).\(settings.format)"
        return outputURL.appendingPathComponent(filename).path
    }

    private func formattedTime(_ seconds: Double?) -> String {
        let totalSeconds = Int(seconds ?? 0)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let remainingSeconds = totalSeconds % 60
        return String(format: "%02d:%02d:%02d", hours, minutes, remainingSeconds)
    }

    private func showAlert(_ message: String, detail: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = detail
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
