import CoreML
import Foundation
import NemotronDiarization

let usage = """
    usage: nemotron-diarize --bundle DIR [options] AUDIO...

    Streams each file through Nemotron-3-Diarization on CoreML, as a live stream fed in --block-ms blocks.

    options:
      --bundle DIR          model bundle (scripts/build_bundle.py)
      --mode MODE           low_latency (default) | very_low_latency | ultra_low_latency
      --compute-units U     gpu (cpu+gpu) | cpu | ane (cpu+neural engine) | all (default: the bundle's)
      --block-ms N          size of the pushed audio blocks (default 80)
      --out DIR             write <DIR>/<file>/{logits,cache_ids,fifo_ids,step_seconds}.npy
      --segments            print speaker segments (threshold 0.5)
      --repeat N            process each file N times, report the best run (default 1)
      --offline             whole-file mode: same results, steps spread over GPU streams + the CPU
      --gpu-streams N       offline: concurrent GPU predictions (default 2)
      --cpu-stride N        offline: every N-th non-pop step on the CPU, 0 = GPU only (default: measured)
      --compute-plan        print where CoreML places the operations of every model function, then exit
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

var bundlePath: String?
var mode = StreamingMode.lowLatency
var units: MLComputeUnits? = nil  // the bundle's default (Neural Engine or GPU)
var blockMs = 80
var outDir: String?
var printSegments = false
var repeats = 1
var files: [String] = []
var offline = false
var printComputePlan = false
var offlineOptions = OfflineDiarizer.Options()
var args = CommandLine.arguments.dropFirst().makeIterator()
while let arg = args.next() {
    switch arg {
    case "--bundle": bundlePath = args.next()
    case "--mode": mode = StreamingMode(rawValue: args.next() ?? "") ?? { fail("unknown mode\n" + usage) }()
    case "--compute-units":
        switch args.next() {
        case "gpu": units = .cpuAndGPU
        case "cpu": units = .cpuOnly
        case "ane": units = .cpuAndNeuralEngine
        case "all": units = .all
        default: fail("unknown compute units\n" + usage)
        }
    case "--block-ms": blockMs = Int(args.next() ?? "") ?? 80
    case "--out": outDir = args.next()
    case "--segments": printSegments = true
    case "--repeat": repeats = max(1, Int(args.next() ?? "") ?? 1)
    case "--offline": offline = true
    case "--compute-plan": printComputePlan = true
    case "--gpu-streams": offlineOptions.gpuStreams = Int(args.next() ?? "") ?? 2
    case "--cpu-stride": offlineOptions.cpuStride = Int(args.next() ?? "")
    case "-h", "--help": print(usage); exit(0)
    default: files.append(arg)
    }
}
guard let bundlePath, !files.isEmpty || printComputePlan else { fail(usage) }

/// Operation counts per preferred compute device, for every function of the bundle's encoder.
func computePlanReport(bundle: URL, units: MLComputeUnits?) async throws {
    let config = try JSONSerialization.jsonObject(with: Data(contentsOf: bundle.appending(path: "config.json")))
        as! [String: Any]
    let layout = config["model_layout"] as! String
    let buckets = (config["buckets"] as? [Int]) ?? [config["max_sequence_length"] as! Int]
    let defaultUnits: MLComputeUnits = layout == "ane" ? .cpuAndNeuralEngine : .cpuAndGPU
    for length in buckets {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units ?? defaultUnits
        if layout == "buckets" || layout == "ane" { configuration.functionName = "len_\(length)" }
        let plan = try await MLComputePlan.load(contentsOf: bundle.appending(path: "Encoder.mlmodelc"),
                                                configuration: configuration)
        guard case .program(let program) = plan.modelStructure,
              let function = program.functions[configuration.functionName ?? "main"] else { continue }
        var counts: [String: Int] = [:]
        for op in function.block.operations where op.operatorName != "const" {
            let device = plan.deviceUsage(for: op)?.preferred
            let name: String
            switch device {
            case .neuralEngine: name = "NeuralEngine"
            case .gpu: name = "GPU"
            case .cpu: name = "CPU"
            default: name = "none"
            }
            counts[name, default: 0] += 1
        }
        print("len_\(length): " + counts.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))
    }
}

if printComputePlan {
    let done = DispatchSemaphore(value: 0)
    Task {
        do { try await computePlanReport(bundle: URL(filePath: bundlePath), units: units) } catch { fail("error: \(error)") }
        done.signal()
    }
    done.wait()
    exit(0)
}

func segments(_ logits: [Float], numFrames: Int, numSpeakers: Int) -> [(Double, Double, Int)] {
    var result: [(Double, Double, Int)] = []
    for spk in 0..<numSpeakers {
        var start: Int?
        for t in 0...numFrames {
            let active = t < numFrames && logits[t * numSpeakers + spk] > 0  // sigmoid > 0.5
            if active && start == nil { start = t }
            if !active, let s = start {
                result.append((Double(s) * 0.01, Double(t) * 0.01, spk))
                start = nil
            }
        }
    }
    return result.sorted { ($0.0, $0.2) < ($1.0, $1.2) }
}

do {
    let loadStart = Date()
    let diarizer = offline
        ? nil : try StreamingDiarizer(bundleURL: URL(filePath: bundlePath), mode: mode, computeUnits: units)
    try diarizer?.warmUp()
    let offlineDiarizer = offline
        ? try OfflineDiarizer(bundleURL: URL(filePath: bundlePath), mode: mode, options: offlineOptions) : nil
    try offlineDiarizer?.warmUp()
    let offlineInfo = offlineDiarizer.map { ", offline: \($0.options.gpuStreams) accelerator streams, CPU stride \($0.cpuStride)" }
    print(String(format: "model loaded in %.2fs (%@%@)", Date().timeIntervalSince(loadStart), mode.rawValue,
                 offlineInfo ?? ""))
    print("file                                    audio_s   proc_s    RTFx  steps  step_ms mean/p50/p95/max")
    var totalAudio = 0.0, totalProc = 0.0
    for path in files {
        let url = URL(filePath: path)
        let audio = try AudioFile.load16kMono(url)
        let block = 16 * blockMs
        var best = Double.infinity
        let numSpeakers = diarizer?.numSpeakers ?? offlineDiarizer!.numSpeakers
        var result = StreamingDiarizer.Output(logits: [], numFrames: 0, numSpeakers: numSpeakers)
        var trace = StreamingDiarizer.Trace()
        for _ in 0..<repeats {
            if let offlineDiarizer {
                let start = DispatchTime.now().uptimeNanoseconds
                let (out, tr) = try offlineDiarizer.run(audio)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if elapsed < best { best = elapsed; result = out; trace = tr }
                continue
            }
            guard let diarizer else { continue }
            diarizer.reset()
            var out = StreamingDiarizer.Output(logits: [], numFrames: 0, numSpeakers: numSpeakers)
            let start = DispatchTime.now().uptimeNanoseconds
            try audio.withUnsafeBufferPointer { samples in
                var i = 0
                while i < samples.count {
                    let r = try diarizer.push(UnsafeBufferPointer(rebasing: samples[i..<min(i + block, samples.count)]))
                    out.logits += r.logits
                    out.numFrames += r.numFrames
                    i += block
                }
            }
            let last = try diarizer.finish()
            out.logits += last.logits
            out.numFrames += last.numFrames
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            if elapsed < best {
                best = elapsed
                result = out
                trace = diarizer.trace
            }
        }
        let duration = Double(audio.count) / 16000
        totalAudio += duration
        totalProc += best
        let ms = trace.stepSeconds.map { $0 * 1000 }.sorted()
        let pct = { (q: Double) in ms.isEmpty ? 0 : ms[min(ms.count - 1, Int(q * Double(ms.count)))] }
        let name = url.deletingPathExtension().lastPathComponent
        let column = String(name.prefix(38)).padding(toLength: 38, withPad: " ", startingAt: 0)
        print(column + String(format: " %8.1f %8.2f %7.1f %6d  %.1f/%.1f/%.1f/%.1f", duration, best, duration / best,
                              ms.count, ms.reduce(0, +) / Double(max(ms.count, 1)), pct(0.5), pct(0.95), ms.last ?? 0))
        if let outDir {
            let dir = URL(filePath: outDir).appending(path: name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Npy.write(result.logits, shape: [result.numFrames, result.numSpeakers],
                          to: dir.appending(path: "logits.npy"))
            try Npy.writePadded(trace.cacheIDs, to: dir.appending(path: "cache_ids.npy"))
            try Npy.writePadded(trace.fifoIDs, to: dir.appending(path: "fifo_ids.npy"))
            try Npy.write(trace.stepSeconds, shape: [trace.stepSeconds.count], to: dir.appending(path: "step_seconds.npy"))
        }
        if printSegments {
            for (s, e, spk) in segments(result.logits, numFrames: result.numFrames, numSpeakers: result.numSpeakers) {
                print(String(format: "  speaker_%d  %8.2f - %8.2f", spk, s, e))
            }
        }
    }
    print(String(format: "overall RTFx %.1f (%.1fs audio in %.2fs)", totalAudio / totalProc, totalAudio, totalProc))
} catch {
    fail("error: \(error)")
}
