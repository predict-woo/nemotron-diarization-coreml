import ArgumentParser
import Foundation
import NemotronDiarization

struct FileCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "file",
        abstract: "Diarize audio files (the default command).",
        discussion: """
            Files are streamed through the model exactly as live audio would be. Any format AVFoundation reads \
            works (wav, m4a, mp3, caf, ...); audio is converted to 16 kHz mono.
            """
    )

    enum Format: String, CaseIterable, ExpressibleByArgument {
        case pretty, rttm, stats

        static var allValueStrings: [String] { [] }  // the help text lists them
    }

    struct Benchmark: ParsableArguments {
        @Flag(help: "Process each file as a whole, running independent steps concurrently. Same results.")
        var offline = false

        @Option(help: ArgumentHelp("Size of the audio blocks pushed to the stream.", valueName: "ms",
                                   visibility: .hidden))
        var blockMs = 80

        @Option(help: ArgumentHelp("Process each file n times and report the fastest run.", valueName: "n",
                                   visibility: .hidden))
        var `repeat` = 1

        @Option(help: ArgumentHelp("Write logits and speaker-cache traces as .npy to <dir>/<file>/.",
                                   valueName: "dir", visibility: .hidden))
        var out: String?

        @Option(help: ArgumentHelp("--offline: concurrent accelerator predictions.", valueName: "n",
                                   visibility: .hidden))
        var gpuStreams = 2

        @Option(help: ArgumentHelp("--offline: every n-th step on the CPU, 0 = none (default: measured).",
                                   valueName: "n", visibility: .hidden))
        var cpuStride: Int?
    }

    @OptionGroup var model: ModelOptions

    @Option(name: .shortAndLong, help: ArgumentHelp(
        "pretty (timeline and speaker turns), rttm (exact segments, standard RTTM) or stats (speed per file).",
        valueName: "format"))
    var format: Format = .pretty

    @OptionGroup var bench: Benchmark

    @Argument(help: ArgumentHelp("Audio files.", valueName: "audio"))
    var files: [String]

    func validate() throws {
        if files.isEmpty { throw ValidationError("give at least one audio file, or try `nemotron-diarize live`") }
        for path in files where !FileManager.default.fileExists(atPath: path) {
            throw ValidationError("no such file: \(path)")
        }
    }

    func run() throws {
        let url = try model.bundleURL()
        var streaming: StreamingDiarizer?
        var offline: OfflineDiarizer?
        try withSpinner("Loading the model") {
            if bench.offline {
                var options = OfflineDiarizer.Options()
                options.gpuStreams = bench.gpuStreams
                options.cpuStride = bench.cpuStride
                offline = try OfflineDiarizer(bundleURL: url, mode: model.latency.mode, options: options)
                try offline?.warmUp()
            } else {
                streaming = try StreamingDiarizer(bundleURL: url, mode: model.latency.mode,
                                                  computeUnits: model.units?.computeUnits)
                try streaming?.warmUp()
            }
        }
        let numSpeakers = streaming?.numSpeakers ?? offline!.numSpeakers

        if format == .stats {
            print(Terminal.dim("\(model.device(url)) · \(model.latency.label)"
                               + (bench.offline ? " · whole-file mode" : "")))
            print(Terminal.bold(pad("file", 28) + "   audio    time    speed  steps   step ms mean / p50 / p95 / max"))
        }
        var totalAudio = 0.0, totalTime = 0.0
        for (index, path) in files.enumerated() {
            let audio = try AudioFile.load16kMono(URL(filePath: path))
            var best = Double.infinity
            var result = StreamingDiarizer.Output(numSpeakers: numSpeakers)
            var trace = StreamingDiarizer.Trace()
            for _ in 0..<max(bench.repeat, 1) {
                let start = DispatchTime.now().uptimeNanoseconds
                let (out, tr) = try offline?.run(audio) ?? stream(audio, through: streaming!)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if elapsed < best { (best, result, trace) = (elapsed, out, tr) }
            }
            let duration = Double(audio.count) / 16000
            totalAudio += duration
            totalTime += best
            let name = URL(filePath: path).deletingPathExtension().lastPathComponent
            if let out = bench.out { try save(result, trace, to: URL(filePath: out).appending(path: name)) }

            switch format {
            case .pretty:
                if index > 0 { print() }
                printPretty(URL(filePath: path).lastPathComponent, result, duration: duration, seconds: best)
            case .rttm:
                for turn in exactSegments(result.logits, numFrames: result.numFrames, numSpeakers: numSpeakers) {
                    print(String(format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>", name, turn.start,
                                 turn.duration, turn.speaker + 1))
                }
            case .stats:
                let ms = trace.stepSeconds.map { $0 * 1000 }.sorted()
                let pct = { (q: Double) in ms.isEmpty ? 0 : ms[min(ms.count - 1, Int(q * Double(ms.count)))] }
                let mean = ms.reduce(0, +) / Double(max(ms.count, 1))
                print(pad(String(name.prefix(28)), 28) + " " + lpad(clock(duration), 7) + String(
                    format: " %6.2f s %7.1f× %6d   %.1f / %.1f / %.1f / %.1f", best, duration / best, ms.count,
                    mean, pct(0.5), pct(0.95), ms.last ?? 0))
            }
        }
        if format == .stats || (format == .pretty && files.count > 1) {
            if format == .pretty { print() }
            print(Terminal.bold(String(format: "Total: %@ of audio in %.2f s · %.1f× real time", clock(totalAudio),
                                       totalTime, totalAudio / totalTime)))
        }
    }

    /// Live-style streaming: the file is pushed in `blockMs` blocks.
    private func stream(_ audio: [Float], through diarizer: StreamingDiarizer) throws
        -> (StreamingDiarizer.Output, StreamingDiarizer.Trace)
    {
        diarizer.reset()
        var out = StreamingDiarizer.Output(numSpeakers: diarizer.numSpeakers)
        let block = max(16 * bench.blockMs, 1)
        try audio.withUnsafeBufferPointer { samples in
            for i in stride(from: 0, to: samples.count, by: block) {
                let r = try diarizer.push(UnsafeBufferPointer(rebasing: samples[i..<min(i + block, samples.count)]))
                out.logits += r.logits
                out.numFrames += r.numFrames
            }
        }
        let last = try diarizer.finish()
        out.logits += last.logits
        out.numFrames += last.numFrames
        return (out, diarizer.trace)
    }

    private func save(_ result: StreamingDiarizer.Output, _ trace: StreamingDiarizer.Trace, to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Npy.write(result.logits, shape: [result.numFrames, result.numSpeakers],
                      to: dir.appending(path: "logits.npy"))
        try Npy.writePadded(trace.cacheIDs, to: dir.appending(path: "cache_ids.npy"))
        try Npy.writePadded(trace.fifoIDs, to: dir.appending(path: "fifo_ids.npy"))
        try Npy.write(trace.stepSeconds, shape: [trace.stepSeconds.count], to: dir.appending(path: "step_seconds.npy"))
    }

    private func printPretty(_ name: String, _ result: StreamingDiarizer.Output, duration: Double, seconds: Double) {
        let tracker = TurnTracker(numSpeakers: result.numSpeakers)
        let turns = tracker.feed(result.logits[...], frames: result.numFrames) + tracker.finish()
        let speakers = Array(Set(turns.map(\.speaker))).sorted()
        let count = speakers.count == 1 ? "1 speaker" : "\(speakers.count) speakers"
        print(Terminal.bold(name) + Terminal.dim(String(format: "  %@ · %@ · diarized in %.2f s (%.0f× real time)",
                                                       clock(duration), count, seconds, duration / seconds)))
        guard !speakers.isEmpty else {
            print(Terminal.dim("  no speech found"))
            return
        }
        print()
        printSpeakerTimeline(result, speakers: speakers, from: 0, to: result.numFrames,
                             axis: (clock(0, decimals: false), clock(duration, decimals: false)))
        print()
        for turn in turns { print(turnLine(turn)) }
    }
}

/// "  0:03.1 – 0:07.9   ● Speaker 1    4.8 s"
func turnLine(_ turn: Turn) -> String {
    "  " + Terminal.dim(lpad(clock(turn.start), 7) + " – " + lpad(clock(turn.end), 7)) + "   "
        + pad(Speaker.label(turn.speaker), 11) + Terminal.dim(String(format: "  %5.1f s", turn.duration))
}

/// Speaker rows over frames [from, to), with talk time and share, and an axis line.
func printSpeakerTimeline(_ result: StreamingDiarizer.Output, speakers: [Int], from: Int, to: Int,
                          axis: (String, String)) {
    let label = 13, stats = 16
    let columns = max(min(Terminal.width - label - stats - 3, 100), 10)
    let window = Array(result.logits[(from * result.numSpeakers)..<(to * result.numSpeakers)])
    let cells = timelineCells(window, firstFrame: from, endFrame: to, numSpeakers: result.numSpeakers,
                              columns: columns, framesPerCell: Double(to - from) / Double(columns))
    var talk = Array(repeating: 0, count: result.numSpeakers)
    for f in from..<to {
        for s in 0..<result.numSpeakers where result.logits[f * result.numSpeakers + s] > 0 { talk[s] += 1 }
    }
    let total = max(speakers.map { talk[$0] }.reduce(0, +), 1)
    for s in speakers {
        print("  " + pad(Speaker.label(s), label) + timelineRow(cells[s], speaker: s)
              + Terminal.dim(String(format: "  %6.1f s %4.0f%%", Double(talk[s]) / 100,
                                    100 * Double(talk[s]) / Double(total))))
    }
    let gap = max(columns - visibleWidth(axis.0) - visibleWidth(axis.1), 1)
    let ticks = axis.0 + String(repeating: " ", count: gap) + axis.1
    print("  " + String(repeating: " ", count: label) + Terminal.dim(ticks))
}
