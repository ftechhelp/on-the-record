import AppKit

struct AppSettings {
    var apiKey: String
    var geminiAPIKey: String
    var outputDirectory: String
    var format: String
    var sourceMode: String
    var chunkSeconds: Int
    var diarize: Bool
    var studyDocEnabled: Bool
    var geminiModel: String

    var includesSystemAudio: Bool {
        sourceMode == "both" || sourceMode == "system"
    }

    var includesMicrophone: Bool {
        sourceMode == "both" || sourceMode == "microphone"
    }
}

final class SettingsWindowController: NSWindowController {
    var onSave: ((AppSettings) -> Void)?
    var onStart: (() -> Void)?
    var onStop: (() -> Void)?
    var onChooseOutputFolder: (() -> Void)?

    private let statusLabel = NSTextField(labelWithString: "Engine not started")
    private let apiKeyField = NSSecureTextField(frame: .zero)
    private let geminiKeyField = NSSecureTextField(frame: .zero)
    private let outputField = NSTextField(frame: .zero)
    private let sourcePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let formatPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let chunkField = NSTextField(frame: .zero)
    private let diarizeCheckbox = NSButton(checkboxWithTitle: "Speaker diarization", target: nil, action: nil)
    private let studyDocCheckbox = NSButton(checkboxWithTitle: "Generate Gemini study document", target: nil, action: nil)
    private let geminiModelField = NSTextField(frame: .zero)
    private let transcriptTextView = NSTextView(frame: .zero)
    private let logTextView = NSTextView(frame: .zero)
    private let startButton = NSButton(title: "Start", target: nil, action: nil)
    private let stopButton = NSButton(title: "Stop", target: nil, action: nil)

    init(settings: AppSettings) {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 720, height: 760),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "On The Record"
        window.minSize = NSSize(width: 600, height: 620)
        super.init(window: window)
        buildInterface()
        apply(settings)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    var currentSettings: AppSettings {
        AppSettings(
            apiKey: apiKeyField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines),
            geminiAPIKey: geminiKeyField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines),
            outputDirectory: (outputField.stringValue as NSString).expandingTildeInPath,
            format: formatPopup.selectedItem?.title.lowercased() ?? "txt",
            sourceMode: selectedSourceMode(),
            chunkSeconds: max(5, chunkField.integerValue),
            diarize: diarizeCheckbox.state == .on,
            studyDocEnabled: studyDocCheckbox.state == .on,
            geminiModel: geminiModelField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "gemini-3-flash-preview"
                : geminiModelField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    func apply(_ settings: AppSettings) {
        apiKeyField.stringValue = settings.apiKey
        geminiKeyField.stringValue = settings.geminiAPIKey
        outputField.stringValue = settings.outputDirectory
        formatPopup.selectItem(withTitle: settings.format.uppercased())
        selectSourceMode(settings.sourceMode)
        chunkField.integerValue = settings.chunkSeconds
        diarizeCheckbox.state = settings.diarize ? .on : .off
        studyDocCheckbox.state = settings.studyDocEnabled ? .on : .off
        geminiModelField.stringValue = settings.geminiModel
    }

    func setStatus(_ text: String, isRecording: Bool) {
        statusLabel.stringValue = text
        startButton.isEnabled = !isRecording
        stopButton.isEnabled = isRecording
    }

    func appendLog(_ message: String) {
        guard !message.isEmpty else { return }
        append(message, to: logTextView)
    }

    func appendTranscript(_ message: String) {
        guard !message.isEmpty else { return }
        append(message, to: transcriptTextView)
    }

    func clearTranscript() {
        transcriptTextView.string = ""
    }

    private func buildInterface() {
        guard let window else { return }

        let rootStack = NSStackView()
        rootStack.orientation = .vertical
        rootStack.alignment = .leading
        rootStack.spacing = 14
        rootStack.edgeInsets = NSEdgeInsets(top: 20, left: 20, bottom: 20, right: 20)
        rootStack.translatesAutoresizingMaskIntoConstraints = false

        let titleLabel = NSTextField(labelWithString: "On The Record")
        titleLabel.font = .systemFont(ofSize: 24, weight: .semibold)
        statusLabel.textColor = .secondaryLabelColor

        rootStack.addArrangedSubview(titleLabel)
        rootStack.addArrangedSubview(statusLabel)

        apiKeyField.placeholderString = "OpenAI API key"
        geminiKeyField.placeholderString = "Gemini API key"
        outputField.placeholderString = "Output folder"
        sourcePopup.addItems(withTitles: ["Both", "System Only", "Microphone Only"])
        formatPopup.addItems(withTitles: ["TXT", "MD", "JSON"])
        chunkField.placeholderString = "15"
        geminiModelField.placeholderString = "gemini-3-flash-preview"

        rootStack.addArrangedSubview(labeledRow("OpenAI API key", apiKeyField))
        rootStack.addArrangedSubview(labeledRow("Gemini API key", geminiKeyField))
        rootStack.addArrangedSubview(labeledRow("Output folder", outputFolderRow()))
        rootStack.addArrangedSubview(labeledRow("Audio source", sourcePopup))
        rootStack.addArrangedSubview(labeledRow("Transcript format", formatPopup))
        rootStack.addArrangedSubview(labeledRow("Chunk seconds", chunkField))
        rootStack.addArrangedSubview(diarizeCheckbox)
        rootStack.addArrangedSubview(studyDocCheckbox)
        rootStack.addArrangedSubview(labeledRow("Gemini model", geminiModelField))

        let buttonRow = NSStackView(views: [saveButton(), startButton, stopButton])
        buttonRow.orientation = .horizontal
        buttonRow.spacing = 8
        rootStack.addArrangedSubview(buttonRow)

        rootStack.addArrangedSubview(sectionLabel("Live transcript"))
        rootStack.addArrangedSubview(scrollView(for: transcriptTextView, height: 180))
        rootStack.addArrangedSubview(sectionLabel("Engine log"))
        rootStack.addArrangedSubview(scrollView(for: logTextView, height: 130))

        let contentView = NSView()
        contentView.addSubview(rootStack)
        window.contentView = contentView
        NSLayoutConstraint.activate([
            rootStack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            rootStack.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            rootStack.topAnchor.constraint(equalTo: contentView.topAnchor),
            rootStack.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
        ])

        startButton.target = self
        startButton.action = #selector(startTapped)
        stopButton.target = self
        stopButton.action = #selector(stopTapped)
        stopButton.isEnabled = false
    }

    private func saveButton() -> NSButton {
        let button = NSButton(title: "Save", target: self, action: #selector(saveTapped))
        button.bezelStyle = .rounded
        return button
    }

    private func outputFolderRow() -> NSView {
        let chooseButton = NSButton(title: "Choose", target: self, action: #selector(chooseOutputTapped))
        chooseButton.bezelStyle = .rounded

        let row = NSStackView(views: [outputField, chooseButton])
        row.orientation = .horizontal
        row.spacing = 8
        outputField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        return row
    }

    private func labeledRow(_ label: String, _ control: NSView) -> NSStackView {
        let labelView = NSTextField(labelWithString: label)
        labelView.alignment = .right
        labelView.widthAnchor.constraint(equalToConstant: 130).isActive = true

        let row = NSStackView(views: [labelView, control])
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 10
        row.widthAnchor.constraint(greaterThanOrEqualToConstant: 520).isActive = true
        control.setContentHuggingPriority(.defaultLow, for: .horizontal)
        return row
    }

    private func sectionLabel(_ text: String) -> NSTextField {
        let label = NSTextField(labelWithString: text)
        label.font = .systemFont(ofSize: 13, weight: .semibold)
        return label
    }

    private func scrollView(for textView: NSTextView, height: CGFloat) -> NSScrollView {
        textView.isEditable = false
        textView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        textView.textContainerInset = NSSize(width: 8, height: 8)

        let scrollView = NSScrollView()
        scrollView.borderType = .bezelBorder
        scrollView.hasVerticalScroller = true
        scrollView.documentView = textView
        scrollView.widthAnchor.constraint(greaterThanOrEqualToConstant: 620).isActive = true
        scrollView.heightAnchor.constraint(equalToConstant: height).isActive = true
        return scrollView
    }

    private func append(_ message: String, to textView: NSTextView) {
        let existing = textView.string
        textView.string = existing.isEmpty ? message : existing + "\n" + message
        textView.scrollToEndOfDocument(nil)
    }

    private func selectedSourceMode() -> String {
        switch sourcePopup.selectedItem?.title {
        case "System Only":
            return "system"
        case "Microphone Only":
            return "microphone"
        default:
            return "both"
        }
    }

    private func selectSourceMode(_ sourceMode: String) {
        switch sourceMode {
        case "system":
            sourcePopup.selectItem(withTitle: "System Only")
        case "microphone":
            sourcePopup.selectItem(withTitle: "Microphone Only")
        default:
            sourcePopup.selectItem(withTitle: "Both")
        }
    }

    @objc private func saveTapped() {
        onSave?(currentSettings)
    }

    @objc private func startTapped() {
        onStart?()
    }

    @objc private func stopTapped() {
        onStop?()
    }

    @objc private func chooseOutputTapped() {
        onChooseOutputFolder?()
    }
}
