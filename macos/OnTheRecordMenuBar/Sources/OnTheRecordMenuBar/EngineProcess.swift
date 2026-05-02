import Foundation

typealias JSONDictionary = [String: Any]

enum EngineProcessError: Error, LocalizedError {
    case alreadyRunning
    case notRunning
    case invalidMessage

    var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "The recording engine is already running."
        case .notRunning:
            return "The recording engine is not running."
        case .invalidMessage:
            return "Could not encode the engine command."
        }
    }
}

final class EngineProcess {
    var onEvent: ((JSONDictionary) -> Void)?
    var onLog: ((String) -> Void)?
    var onExit: ((Int32) -> Void)?

    private var process: Process?
    private var inputPipe: Pipe?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?
    private var outputBuffer = Data()
    private let parseQueue = DispatchQueue(label: "on-the-record.engine.parse")

    var isRunning: Bool {
        process?.isRunning == true
    }

    func start(
        executablePath: String,
        arguments: [String],
        workingDirectory: URL
    ) throws {
        guard process?.isRunning != true else {
            throw EngineProcessError.alreadyRunning
        }

        let process = Process()
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()

        process.executableURL = URL(fileURLWithPath: executablePath)
        process.arguments = arguments
        process.currentDirectoryURL = workingDirectory
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe

        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.consumeOutput(data)
        }

        errorPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else {
                return
            }
            self?.onLog?(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        process.terminationHandler = { [weak self] terminatedProcess in
            outputPipe.fileHandleForReading.readabilityHandler = nil
            errorPipe.fileHandleForReading.readabilityHandler = nil
            self?.onExit?(terminatedProcess.terminationStatus)
        }

        try process.run()

        self.process = process
        self.inputPipe = inputPipe
        self.outputPipe = outputPipe
        self.errorPipe = errorPipe
    }

    func send(command: String, payload: JSONDictionary = [:]) throws {
        guard process?.isRunning == true, let inputPipe else {
            throw EngineProcessError.notRunning
        }

        var message: JSONDictionary = [
            "id": UUID().uuidString,
            "command": command,
        ]
        if !payload.isEmpty {
            message["payload"] = payload
        }

        guard JSONSerialization.isValidJSONObject(message),
              let data = try? JSONSerialization.data(withJSONObject: message) else {
            throw EngineProcessError.invalidMessage
        }

        inputPipe.fileHandleForWriting.write(data)
        inputPipe.fileHandleForWriting.write(Data("\n".utf8))
    }

    func stop() {
        try? send(command: "shutdown")
        inputPipe?.fileHandleForWriting.closeFile()
        process?.terminate()
        process = nil
        inputPipe = nil
        outputPipe = nil
        errorPipe = nil
        outputBuffer.removeAll()
    }

    private func consumeOutput(_ data: Data) {
        parseQueue.async { [weak self] in
            guard let self else { return }
            self.outputBuffer.append(data)

            while let newlineIndex = self.outputBuffer.firstIndex(of: 10) {
                let lineData = self.outputBuffer[..<newlineIndex]
                self.outputBuffer.removeSubrange(...newlineIndex)

                guard !lineData.isEmpty else { continue }
                self.parseLine(Data(lineData))
            }
        }
    }

    private func parseLine(_ data: Data) {
        do {
            let object = try JSONSerialization.jsonObject(with: data)
            if let message = object as? JSONDictionary {
                onEvent?(message)
            }
        } catch {
            let text = String(data: data, encoding: .utf8) ?? "<invalid utf8>"
            onLog?("Could not parse engine output: \(text)")
        }
    }
}
