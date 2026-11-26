import ArgumentParser
import CoreML
import Foundation
import NemotronDiarization

@main
struct NemotronDiarize: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "nemotron-diarize",
        abstract: "Speaker diarization with Nemotron-3-Diarization on the Apple Neural Engine.",
        discussion: """
            Examples:
              nemotron-diarize live                  # the microphone, live
              nemotron-diarize live --file talk.m4a  # a recording, in real time
              nemotron-diarize meeting.wav           # diarize files
              nemotron-diarize meeting.wav -f rttm   # exact segments as RTTM
            """,
        subcommands: [FileCommand.self, LiveCommand.self, PlanCommand.self],
        defaultSubcommand: FileCommand.self
    )
}

// MARK: - shared options

enum Latency: String, CaseIterable, ExpressibleByArgument {
    case low
    case veryLow = "very-low"
    case ultraLow = "ultra-low"

    static var allValueStrings: [String] { [] }  // the help text lists them

    init?(argument: String) {
        let key = argument.lowercased().replacingOccurrences(of: "_", with: "-")
        let match = Latency.allCases.first { key == $0.rawValue || key == $0.rawValue + "-latency" }
        guard let match else { return nil }
        self = match
    }

    var mode: StreamingMode {
        switch self {
        case .low: .lowLatency
        case .veryLow: .veryLowLatency
        case .ultraLow: .ultraLowLatency
        }
    }

    /// Model-card latency: chunk + look-ahead.
    var seconds: Double { Double(mode.chunkLength + mode.rightContext) * 0.08 }
    var label: String {
        String(format: "%@ latency (%.2f s)", rawValue.replacingOccurrences(of: "-", with: " "), seconds)
    }
}

enum Units: String, CaseIterable, ExpressibleByArgument {
    case ane, gpu, cpu, all

    var computeUnits: MLComputeUnits {
        switch self {
        case .ane: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        case .cpu: .cpuOnly
        case .all: .all
        }
    }
}

struct ModelOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: ArgumentHelp(
        "Model bundle. Default: $NEMOTRON_BUNDLE, else ./models/Nemotron3Diarization.", valueName: "dir"))
    var bundle: String?

    @Option(name: .shortAndLong, help: ArgumentHelp(
        "How far results trail the audio: low (1.04 s, most accurate), very-low (0.64 s) or ultra-low (0.32 s).",
        valueName: "latency"))
    var latency: Latency = .low

    @Option(name: .customLong("compute-units"), help: ArgumentHelp(
        "Override where the model runs: ane, gpu, cpu or all. Default: what the bundle is built for.",
        valueName: "units", visibility: .hidden))
    var units: Units?

    static let defaultBundle = "models/Nemotron3Diarization"

    func bundleURL() throws -> URL {
        let path = bundle ?? ProcessInfo.processInfo.environment["NEMOTRON_BUNDLE"] ?? Self.defaultBundle
        let url = URL(filePath: path)
        guard FileManager.default.fileExists(atPath: url.appending(path: "config.json").path) else {
            if bundle == nil {
                throw CLIError("""
                    no model bundle at \(path). Download it with
                      hf download predict-woo/Nemotron-3-Diarization-CoreML --include "Nemotron3Diarization/*" --local-dir models
                    or pass --bundle <dir>.
                    """)
            }
            throw CLIError("\(path) is not a model bundle (no config.json)")
        }
        return url
    }

    /// "Neural Engine" or "GPU": where the model runs.
    func device(_ url: URL) -> String {
        if let units { return units == .ane ? "Neural Engine" : units.rawValue.uppercased() }
        let config = (try? JSONSerialization.jsonObject(with: Data(contentsOf: url.appending(path: "config.json"))))
            as? [String: Any]
        return config?["model_layout"] as? String == "ane" ? "Neural Engine" : "GPU"
    }

    /// Loads and warms up the streaming diarizer, with a progress line on the terminal.
    func loadDiarizer() throws -> (StreamingDiarizer, url: URL) {
        let url = try bundleURL()
        let diarizer = try withSpinner("Loading the model") {
            let diarizer = try StreamingDiarizer(bundleURL: url, mode: latency.mode, computeUnits: units?.computeUnits)
            try diarizer.warmUp()
            return diarizer
        }
        return (diarizer, url)
    }
}

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

// MARK: - progress

/// Runs `work` while a spinner with the elapsed time runs on stderr (terminals only). A first launch compiles the
/// model for the Neural Engine, which takes about a minute; the spinner says so once loading takes a while.
func withSpinner<T>(_ title: String, _ work: () throws -> T) rethrows -> T {
    guard isatty(STDERR_FILENO) != 0 else { return try work() }
    let frames = Array("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    let start = Date()
    let queue = DispatchQueue(label: "spinner")
    let timer = DispatchSource.makeTimerSource(queue: queue)
    var tick = 0
    timer.schedule(deadline: .now(), repeating: .milliseconds(80))
    timer.setEventHandler {
        let elapsed = Date().timeIntervalSince(start)
        var line = Terminal.style(String(frames[tick % frames.count]), "36") + " " + title
            + Terminal.dim(String(format: "  %.0f s", elapsed))
        if elapsed > 4 {
            line += Terminal.dim("  · first launch: compiling for this Mac, about a minute (cached afterwards)")
        }
        FileHandle.standardError.write(Data(("\r\u{1B}[2K" + line).utf8))
        tick += 1
    }
    timer.resume()
    defer {
        timer.cancel()
        queue.sync {}  // a tick in flight finishes before the line is cleared
        FileHandle.standardError.write(Data("\r\u{1B}[2K".utf8))
    }
    return try work()
}
