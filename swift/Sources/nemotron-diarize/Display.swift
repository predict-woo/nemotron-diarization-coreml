import Foundation

// MARK: - terminal

/// Styling that degrades to plain text when the output is not a terminal (or NO_COLOR is set).
enum Terminal {
    static let isTTY = isatty(STDOUT_FILENO) != 0
    nonisolated(unsafe) static var color: Bool = {
        let env = ProcessInfo.processInfo.environment
        if let force = env["CLICOLOR_FORCE"], force != "0" { return true }
        return isTTY && env["NO_COLOR"] == nil
    }()
    nonisolated(unsafe) private static var savedMode: termios?

    /// Terminal width; when the output is not a terminal, $COLUMNS or 100.
    static var width: Int {
        var size = winsize()
        if isTTY, ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0, size.ws_col > 0 { return Int(size.ws_col) }
        return ProcessInfo.processInfo.environment["COLUMNS"].flatMap(Int.init) ?? 100
    }

    static func style(_ text: String, _ code: String) -> String {
        color ? "\u{1B}[\(code)m\(text)\u{1B}[0m" : text
    }

    static func bold(_ text: String) -> String { style(text, "1") }
    static func dim(_ text: String) -> String { style(text, "2") }
    static func rgb(_ text: String, _ c: (Int, Int, Int)) -> String { style(text, "38;2;\(c.0);\(c.1);\(c.2)") }

    /// Writes to stdout through the same buffer as `print`, and flushes.
    static func write(_ text: String) {
        fputs(text, stdout)
        fflush(stdout)
    }

    /// For a live view: hides the cursor and stops typed keys from echoing (restored at exit).
    static func hideCursor() {
        write("\u{1B}[?25l")
        var mode = termios()
        if isatty(STDIN_FILENO) != 0, tcgetattr(STDIN_FILENO, &mode) == 0 {
            savedMode = mode
            mode.c_lflag &= ~tcflag_t(ECHO | ICANON)
            tcsetattr(STDIN_FILENO, TCSANOW, &mode)
        }
        atexit { Terminal.showCursor() }
    }

    static func showCursor() {
        write("\u{1B}[?25h")
        if var mode = savedMode {
            tcsetattr(STDIN_FILENO, TCSANOW, &mode)
            savedMode = nil
        }
    }

    /// A status line that the next `status` or `clearStatus` call overwrites (terminals only).
    static func status(_ text: String) {
        if isTTY { write("\r\u{1B}[2K" + text) }
    }

    static func clearStatus() {
        if isTTY { write("\r\u{1B}[2K") }
    }
}

/// Visible width of a string with ANSI styles.
func visibleWidth(_ text: String) -> Int {
    var width = 0, inEscape = false
    for scalar in text.unicodeScalars {
        if inEscape {
            if scalar == "m" { inEscape = false }
        } else if scalar == "\u{1B}" {
            inEscape = true
        } else {
            width += 1
        }
    }
    return width
}

func pad(_ text: String, _ width: Int) -> String {
    text + String(repeating: " ", count: max(0, width - visibleWidth(text)))
}

func lpad(_ text: String, _ width: Int) -> String {
    String(repeating: " ", count: max(0, width - visibleWidth(text))) + text
}

// MARK: - speakers

enum Speaker {
    /// Categorical palette (dark-surface steps), in a colour-vision-deficiency-safe order.
    static let colors: [(Int, Int, Int)] = [
        (0x33, 0x8C, 0xED), (0xE7, 0x66, 0x24), (0x00, 0xA8, 0x88), (0xD2, 0x8E, 0x05),
        (0xE2, 0x4D, 0x8B), (0x0A, 0xAC, 0x23), (0x98, 0x87, 0xEF), (0xEB, 0x62, 0x63),
    ]

    static func name(_ k: Int) -> String { "Speaker \(k + 1)" }
    static func paint(_ text: String, _ k: Int) -> String { Terminal.rgb(text, colors[k % colors.count]) }
    static func label(_ k: Int) -> String { paint("●", k) + " " + name(k) }
}

/// 1:02.3, or 1:02:03 past an hour.
func clock(_ seconds: Double, decimals: Bool = true) -> String {
    let s = max(seconds, 0)
    if s >= 3600 {
        return String(format: "%d:%02d:%02d", Int(s) / 3600, Int(s) / 60 % 60, Int(s) % 60)
    }
    return decimals ? String(format: "%d:%04.1f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60))
        : String(format: "%d:%02d", Int(s) / 60, Int(s) % 60)
}

// MARK: - turns

struct Turn {
    let speaker: Int
    let start: Double
    let end: Double
    var duration: Double { end - start }
}

/// Speaker turns from the 10 ms speaker probabilities, smoothed for reading: pauses up to `bridge` seconds are
/// bridged and turns shorter than `minimum` are dropped. (`exactSegments` gives the unsmoothed segments.)
final class TurnTracker {
    let numSpeakers: Int
    let bridgeFrames: Int
    let minimumFrames: Int
    private var openStart: [Int?]
    private var lastActive: [Int]
    private(set) var frames = 0

    init(numSpeakers: Int, bridge: Double = 1.0, minimum: Double = 0.25) {
        self.numSpeakers = numSpeakers
        bridgeFrames = Int(bridge * 100)
        minimumFrames = Int(minimum * 100)
        openStart = Array(repeating: nil, count: numSpeakers)
        lastActive = Array(repeating: -1, count: numSpeakers)
    }

    /// Feeds frames of logits (row-major frames x speakers); returns the turns that ended, in start order.
    func feed(_ logits: ArraySlice<Float>, frames n: Int) -> [Turn] {
        var ended: [Turn] = []
        let base = logits.startIndex
        for i in 0..<n {
            let frame = frames + i
            for s in 0..<numSpeakers {
                if logits[base + i * numSpeakers + s] > 0 {  // sigmoid > 0.5
                    if openStart[s] == nil { openStart[s] = frame }
                    lastActive[s] = frame
                } else if let start = openStart[s], frame - lastActive[s] > bridgeFrames {
                    if let turn = close(s, start) { ended.append(turn) }
                }
            }
        }
        frames += n
        return ended.sorted { $0.start < $1.start }
    }

    /// Closes every open turn (end of the stream).
    func finish() -> [Turn] {
        (0..<numSpeakers).compactMap { s in openStart[s].flatMap { close(s, $0) } }.sorted { $0.start < $1.start }
    }

    /// Speakers active within the last `window` seconds of the frames fed so far.
    func speaking(within window: Double = 0.5) -> [Int] {
        (0..<numSpeakers).filter { openStart[$0] != nil && frames - lastActive[$0] <= Int(window * 100) }
    }

    private func close(_ s: Int, _ start: Int) -> Turn? {
        openStart[s] = nil
        let end = lastActive[s] + 1
        return end - start >= minimumFrames ? Turn(speaker: s, start: Double(start) / 100, end: Double(end) / 100)
            : nil
    }
}

/// Unsmoothed segments, exactly the model's thresholded output (as the HF processor's extract_speaker_dict).
func exactSegments(_ logits: [Float], numFrames: Int, numSpeakers: Int) -> [Turn] {
    var result: [Turn] = []
    for s in 0..<numSpeakers {
        var start: Int?
        for t in 0...numFrames {
            let active = t < numFrames && logits[t * numSpeakers + s] > 0
            if active && start == nil { start = t }
            if !active, let begin = start {
                result.append(Turn(speaker: s, start: Double(begin) / 100, end: Double(t) / 100))
                start = nil
            }
        }
    }
    return result.sorted { ($0.start, $0.speaker) < ($1.start, $1.speaker) }
}

// MARK: - timeline

/// One row per speaker over `columns` cells, each `framesPerCell` 10 ms frames wide (the last cells end at
/// `endFrame`); a cell is filled where the speaker is active in at least 30% of its frames.
func timelineCells(_ logits: [Float], firstFrame: Int, endFrame: Int, numSpeakers: Int, columns: Int,
                   framesPerCell: Double) -> [[Bool]] {
    var rows = Array(repeating: Array(repeating: false, count: columns), count: numSpeakers)
    let available = logits.count / numSpeakers
    for c in 0..<columns {
        let hi = Double(endFrame) - Double(columns - 1 - c) * framesPerCell
        let lo = hi - framesPerCell
        let a = max(Int(lo.rounded()) - firstFrame, 0), b = min(Int(hi.rounded()) - firstFrame, available)
        guard b > a else { continue }
        for s in 0..<numSpeakers {
            var active = 0
            for f in a..<b where logits[f * numSpeakers + s] > 0 { active += 1 }
            rows[s][c] = Double(active) >= 0.3 * Double(b - a)
        }
    }
    return rows
}

func timelineRow(_ cells: [Bool], speaker: Int) -> String {
    var row = "", run = ""
    var filled: Bool?
    for cell in cells {
        if cell != filled, !run.isEmpty {
            row += filled == true ? Speaker.paint(run, speaker) : Terminal.dim(run)
            run = ""
        }
        filled = cell
        run += cell ? "█" : "·"
    }
    if !run.isEmpty { row += filled == true ? Speaker.paint(run, speaker) : Terminal.dim(run) }
    return row
}
