import AVFoundation
import ArgumentParser
import Foundation
import NemotronDiarization

struct LiveCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "live",
        abstract: "Diarize the microphone live: who is speaking, as they speak.",
        discussion: """
            Shows a scrolling timeline per speaker and logs each speaker turn as it ends. Results trail the audio \
            by the latency (1.04 s in low latency mode, plus up to 0.72 s until the next chunk completes). \
            Press Ctrl-C to stop and see the summary. Speakers are numbered in order of first appearance.

            The first run asks for microphone access for your terminal app.
            """
    )

    @OptionGroup var model: ModelOptions

    @Option(help: ArgumentHelp("Play an audio file in real time instead of listening to the microphone.",
                               valueName: "audio"))
    var file: String?

    @Option(help: ArgumentHelp("Playback speed with --file.", valueName: "x"))
    var speed: Double = 1

    @Option(help: ArgumentHelp("Stop after this many seconds of audio.", valueName: "seconds"))
    var duration: Double?

    @Option(help: ArgumentHelp("Seconds of history in the timeline.", valueName: "seconds"))
    var window: Double = 30

    func validate() throws {
        if let file, !FileManager.default.fileExists(atPath: file) { throw ValidationError("no such file: \(file)") }
        if speed <= 0 { throw ValidationError("--speed must be positive") }
        if window < 5 { throw ValidationError("--window must be at least 5 seconds") }
    }

    func run() throws {
        let (diarizer, url) = try model.loadDiarizer()
        diarizer.recordTrace = false
        let session = LiveSession(diarizer: diarizer, window: window, limit: duration, isMicrophone: file == nil)
        let detail = [model.latency.label, model.device(url)]

        if let file {
            let audio = try AudioFile.load16kMono(URL(filePath: file))
            let player = FilePlayer(audio: audio, speed: speed)
            let pace = speed == 1 ? "in real time" : String(format: "at %g× speed", speed)
            session.total = Double(audio.count) / 16000
            session.start(source: "Playing \(URL(filePath: file).lastPathComponent) \(pace)", detail: detail)
            session.onStop = { player.stop() }
            player.start { samples, end in
                session.process(samples)
                if end { session.stop() }
            }
        } else {
            try Microphone.authorize()
            let microphone = Microphone()
            session.start(source: "Listening to \(Microphone.deviceName)", detail: detail)
            session.onStop = { microphone.stop() }
            try microphone.start { samples in DispatchQueue.main.async { session.process(samples) } }
        }

        for sig in [SIGINT, SIGTERM] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { session.stop() }
            source.resume()
            signalSources.append(source)
        }
        dispatchMain()
    }
}

nonisolated(unsafe) private var signalSources: [DispatchSourceSignal] = []

// MARK: - live view

/// Runs on the main queue: pushes audio, keeps the speaker state and draws the view.
final class LiveSession {
    let diarizer: StreamingDiarizer
    let numSpeakers: Int
    let window: Double
    let limit: Double?
    let isMicrophone: Bool
    var total: Double?
    var onStop: (() -> Void)?

    private let tracker: TurnTracker
    private var recent: [Float] = []  // logits of the frames from `recentStart` on
    private var recentStart = 0
    private var frames = 0
    private var talk: [Int]
    private var seen: [Int] = []  // speakers in order of appearance
    private var samples = 0
    private var levels: [Float] = []
    private var stepMs: [Double] = []  // recent step times
    private var pending: [String] = []
    private var liveLines = 0
    private var stopped = false
    private let tty = Terminal.isTTY

    init(diarizer: StreamingDiarizer, window: Double, limit: Double?, isMicrophone: Bool) {
        self.diarizer = diarizer
        numSpeakers = diarizer.numSpeakers
        self.window = window
        self.limit = limit
        self.isMicrophone = isMicrophone
        tracker = TurnTracker(numSpeakers: numSpeakers)
        talk = Array(repeating: 0, count: numSpeakers)
    }

    func start(source: String, detail: [String]) {
        print("  " + Terminal.bold("Nemotron-3-Diarization") + Terminal.dim(" · live"))
        print("  " + source + Terminal.dim(" · " + detail.joined(separator: " · ")))
        print()
        if tty {
            Terminal.hideCursor()
            render()
        }
    }

    func process(_ block: [Float]) {
        guard !stopped else { return }
        var block = block[...]
        if let limit {
            block = block.prefix(max(Int(limit * 16000) - samples, 0))
        }
        samples += block.count
        levels = (levels + [level(block)]).suffix(8)
        let t0 = DispatchTime.now().uptimeNanoseconds
        do {
            let out = try diarizer.push(Array(block))
            if out.numFrames > 0 {
                stepMs = (stepMs + [Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6]).suffix(8)
            }
            absorb(out)
        } catch {
            fail(error)
        }
        if let limit, Double(samples) >= limit * 16000 { return stop() }
        render()
    }

    func stop() {
        guard !stopped else { return }
        stopped = true
        onStop?()
        do { absorb(try diarizer.finish()) } catch { fail(error) }
        pending += tracker.finish().map(turnLine)
        clearLive()
        for line in pending { print(line) }
        pending = []
        if tty { Terminal.showCursor() }
        summary()
        exit(0)
    }

    // MARK: state

    private func absorb(_ out: StreamingDiarizer.Output) {
        guard out.numFrames > 0 else { return }
        recent += out.logits
        for i in 0..<out.numFrames {
            for s in 0..<numSpeakers where out.logits[i * numSpeakers + s] > 0 {
                talk[s] += 1
                if talk[s] == 25 { seen.append(s) }  // 0.25 s of speech gives the speaker a row
            }
        }
        frames += out.numFrames
        pending += tracker.feed(out.logits[...], frames: out.numFrames).map(turnLine)
        let keep = Int(window * 100) + 100  // frames of history for the timeline
        if recent.count / numSpeakers > 2 * keep {
            let drop = recent.count / numSpeakers - keep
            recent.removeFirst(drop * numSpeakers)
            recentStart += drop
        }
    }

    private func level(_ block: ArraySlice<Float>) -> Float {
        guard !block.isEmpty else { return -90 }
        let power = block.reduce(0) { $0 + $1 * $1 } / Float(block.count)
        return 10 * log10(max(power, 1e-9))
    }

    // MARK: drawing

    private func render() {
        guard tty else {  // plain output: the turns, as they end
            for line in pending { print(line) }
            pending = []
            fflush(stdout)
            return
        }
        var text = clearSequence()
        for line in pending { text += line + "\n" }
        pending = []
        let lines = liveRegion()
        text += lines.joined(separator: "\n")
        liveLines = lines.count
        Terminal.write(text)
    }

    private func clearSequence() -> String {
        guard liveLines > 0 else { return "" }
        return "\r" + (liveLines > 1 ? "\u{1B}[\(liveLines - 1)A" : "") + "\u{1B}[J"
    }

    private func clearLive() {
        if tty { Terminal.write(clearSequence()) }
        liveLines = 0
    }

    private func liveRegion() -> [String] {
        let width = Terminal.width
        let label = 13
        let columns = max(min(width - label - 4, 120), 10)
        var lines = [""]
        if seen.isEmpty {
            lines.append("  " + Terminal.dim(frames == 0 ? "Starting…" : "Waiting for speech…"))
        } else {
            let cells = timelineCells(recent, firstFrame: recentStart, endFrame: frames, numSpeakers: numSpeakers,
                                      columns: columns, framesPerCell: window * 100 / Double(columns))
            let speaking = Set(tracker.speaking(within: 0.3))
            for s in seen {
                let name = speaking.contains(s) ? Terminal.bold(Speaker.name(s)) : Speaker.name(s)
                let legend = pad(Speaker.paint("●", s) + " " + name, label)
                lines.append("  " + legend + timelineRow(cells[s], speaker: s))
            }
            let left = String(format: "−%g s", window), right = "now"
            lines.append("  " + String(repeating: " ", count: label) + Terminal.dim(
                left + String(repeating: " ", count: max(columns - left.count - right.count, 1)) + right))
        }
        lines.append("")
        lines.append(statusLine(width: width))
        return lines
    }

    private func statusLine(width: Int) -> String {
        let now = Double(samples) / 16000
        var head: String
        if isMicrophone {
            head = Terminal.style("●", "31") + " " + Terminal.bold(clock(now))
        } else {
            head = "▶ " + Terminal.bold(clock(now)) + (total.map { Terminal.dim(" / " + clock($0)) } ?? "")
        }
        let bars = Array("▁▂▃▄▅▆▇█")
        let meter = levels.map { db in bars[min(max(Int((db + 60) / 50 * 8), 0), 7)] }
        head += "  " + Terminal.style(String(meter), "36")
        var parts = [head]
        if !stepMs.isEmpty {  // median of the recent steps
            parts.append(Terminal.dim(String(format: "%.0f ms per step", stepMs.sorted()[stepMs.count / 2])))
        }
        if frames > 0 { parts.append(Terminal.dim(String(format: "%.1f s behind", now - Double(frames) / 100))) }
        parts.append(Terminal.dim("Ctrl-C to stop"))
        while parts.count > 1 && visibleWidth("  " + parts.joined(separator: "   ")) >= width {
            parts.remove(at: parts.count - 2)
        }
        return "  " + parts.joined(separator: "   ")
    }

    private func summary() {
        let speakers = seen
        let heard = Double(samples) / 16000
        print()
        let count = speakers.count == 1 ? "1 speaker" : "\(speakers.count) speakers"
        print("  " + Terminal.bold("Summary") + Terminal.dim(" · \(clock(heard)) of audio · \(count)"))
        guard !speakers.isEmpty else { return }
        let totalTalk = max(speakers.map { talk[$0] }.reduce(0, +), 1)
        let longest = max(speakers.map { talk[$0] }.max() ?? 1, 1)
        let barWidth = max(min(Terminal.width - 40, 40), 10)
        for s in speakers {
            let share = Double(talk[s]) / Double(totalTalk)
            let length = Double(talk[s]) / Double(longest) * Double(barWidth)
            let bar = String(repeating: "█", count: max(Int(length.rounded()), 1))
            print("  " + pad(Speaker.label(s), 13) + Speaker.paint(pad(bar, barWidth), s)
                  + Terminal.dim(String(format: "  %6.1f s %4.0f%%", Double(talk[s]) / 100, share * 100)))
        }
    }

    private func fail(_ error: Error) -> Never {
        clearLive()
        if tty { Terminal.showCursor() }
        FileHandle.standardError.write(Data("Error: \(error)\n".utf8))
        exit(1)
    }
}

// MARK: - audio sources

/// The default input device, converted to 16 kHz mono.
final class Microphone {
    private let engine = AVAudioEngine()

    static var deviceName: String {
        AVCaptureDevice.default(for: .audio)?.localizedName ?? "the default microphone"
    }

    static func authorize() throws {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return
        case .notDetermined:
            let answered = DispatchSemaphore(value: 0)
            nonisolated(unsafe) var granted = false
            AVCaptureDevice.requestAccess(for: .audio) {
                granted = $0
                answered.signal()
            }
            answered.wait()
            if granted { return }
        default:
            break
        }
        throw CLIError("""
            microphone access is off for this terminal app. Turn it on in System Settings › Privacy & Security › \
            Microphone, then run again.
            """)
    }

    func start(_ handler: @escaping ([Float]) -> Void) throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw CLIError("no microphone found") }
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: format, to: target) else {
            throw CLIError("cannot convert microphone audio (\(format)) to 16 kHz mono")
        }
        converter.downmix = true
        let ratio = 16000 / format.sampleRate
        input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(format.sampleRate / 20), format: format) { buffer, _ in
            guard let out = AVAudioPCMBuffer(pcmFormat: target,
                                             frameCapacity: AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64)
            else { return }
            var supplied = false
            var error: NSError?
            let status = converter.convert(to: out, error: &error) { _, inputStatus in
                if supplied {
                    inputStatus.pointee = .noDataNow
                    return nil
                }
                supplied = true
                inputStatus.pointee = .haveData
                return buffer
            }
            guard status != .error, out.frameLength > 0 else { return }
            handler(Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength))))
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

/// Feeds a recording at its real-time pace (or `speed` times faster), on the main queue.
final class FilePlayer {
    let audio: [Float]
    let speed: Double
    private var position = 0
    private var timer: DispatchSourceTimer?

    init(audio: [Float], speed: Double) {
        self.audio = audio
        self.speed = speed
    }

    func start(_ handler: @escaping (_ samples: [Float], _ end: Bool) -> Void) {
        let start = DispatchTime.now().uptimeNanoseconds
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now(), repeating: .milliseconds(20))
        timer.setEventHandler { [self] in
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            let due = min(Int(elapsed * speed * 16000), audio.count)
            guard due > position else { return }
            let block = Array(audio[position..<due])
            position = due
            handler(block, due == audio.count)
        }
        timer.resume()
        self.timer = timer
    }

    func stop() {
        timer?.cancel()
    }
}
