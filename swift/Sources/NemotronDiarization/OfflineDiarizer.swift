import CoreML
import Foundation

/// Whole-recording diarization that produces the results of `StreamingDiarizer` (the same streaming steps, the
/// same speaker-cache trajectory), using the GPU and the CPU concurrently.
///
/// A streaming step depends on earlier steps only through the speaker cache, and the cache changes only at
/// "pop" steps, which are fixed by the chunk schedule (FIFO lengths do not depend on the audio). So once the
/// cache after a pop is known, every step up to the next pop can run, in any order, on any device:
///   - pop steps form the critical chain; they always run on the GPU, first in line;
///   - the other steps run on `gpuStreams` concurrent GPU streams sharing one copy of the GPU model, and every
///     `cpuStride`-th of them on the CPU (CoreML CPU model). The assignment is static, so results are
///     deterministic. CPU and GPU agree to fp32 rounding (same identity criteria as the streaming runtime).
public final class OfflineDiarizer {
    public struct Options {
        /// Concurrent GPU predictions (they share one copy of the GPU model).
        public var gpuStreams = 2
        /// Every `cpuStride`-th non-pop step runs on the CPU; 0 disables the CPU stream, nil sizes the CPU share
        /// from the measured CPU / GPU step times at `warmUp()` (7 on an M2 Pro).
        public var cpuStride: Int? = nil
        public init() {}
    }

    public let mode: StreamingMode
    public let numSpeakers: Int
    public let options: Options
    /// CPU share in use: every `cpuStride`-th non-pop step (0: GPU only).
    public private(set) var cpuStride: Int

    private let bundle: ModelBundle
    private let embedder: Embedder
    private let gpuRunners: [EncoderRunner]
    private let cpuRunner: EncoderRunner?
    /// fp32 runner for the pop steps (Neural Engine bundles with an exact model): they run on it, in order
    private let exactRunner: EncoderRunner?
    private let hidden: Int
    private let factor: Int

    public init(bundleURL: URL, mode: StreamingMode = .lowLatency, options: Options = Options()) throws {
        bundle = try ModelBundle(url: bundleURL)
        let cfg = bundle.config
        self.mode = mode
        self.options = options
        hidden = cfg.hiddenSize
        factor = cfg.subsamplingFactor
        numSpeakers = cfg.numSpeakers
        embedder = Embedder(weight: bundle.stackingWeight, hidden: hidden, factor: factor)
        // a Neural Engine bundle runs on the ANE streams only: its float16 graph is slow on the CPU
        let isANE = bundle.config.modelLayout == "ane"
        cpuStride = options.cpuStride ?? (isANE ? 0 : 7)
        // GPU and CPU models load concurrently; the CPU takes a small share of the steps, so its largest bucket
        // alone is enough (one weight copy)
        let bundle = self.bundle, useCPU = cpuStride > 0
        var gpu: Result<EncoderModels, Error>?, cpu: Result<EncoderModels, Error>?
        DispatchQueue.concurrentPerform(iterations: useCPU ? 2 : 1) { i in
            if i == 0 {
                gpu = Result { try bundle.encoderModels(computeUnits: bundle.defaultComputeUnits) }
            } else {
                cpu = Result { try bundle.encoderModels(computeUnits: .cpuOnly, buckets: []) }
            }
        }
        let gpuModels = try gpu!.get()
        exactRunner = try bundle.exactEncoderModels().map { try EncoderRunner(models: $0) }
        gpuRunners = try (0..<max(1, options.gpuStreams)).map { _ in try EncoderRunner(models: gpuModels) }
        cpuRunner = try cpu.map { try EncoderRunner(models: try $0.get()) }
    }

    public func warmUp() throws {
        for runner in gpuRunners { try runner.warmUp() }
        try exactRunner?.warmUp()
        guard let cpuRunner else { return }
        try cpuRunner.warmUp()
        if options.cpuStride == nil {
            // the CPU should finish its share no later than the GPU streams finish theirs: with 2 GPU streams on
            // an M2 Pro, a CPU step takes ~3.4 GPU steps and every 7th step is the best share
            let length = cpuRunner.maxLength
            func median(_ runner: EncoderRunner) throws -> Double {
                var times: [Double] = []
                for _ in 0..<3 {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    _ = try runner.run(length: length)
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - t0))
                }
                return times.sorted()[1]
            }
            let ratio = try median(cpuRunner) / median(gpuRunners[0])
            cpuStride = max(3, Int((2 * ratio).rounded()))
        }
    }

    // MARK: plan

    private struct Step {
        let spec: ChunkSpec
        var numFrames = 0  // mel frames of the chunk
        var embeds: [Float] = []  // (numEmbeds x hidden) feature-stacking embeddings
        var numEmbeds = 0
        var numChunk = 0  // scored frames (pushed to the FIFO)
        var firstID = 0  // global encoder frame id of the first chunk frame
        var numCache = 0  // cache / FIFO lengths before the step
        var numFifo = 0
        var epoch = 0  // cache after `epoch` pops
        var isPop = false
        var outOffset = 0  // first output row (10 ms frames)
        var onCPU = false
        var onExact = false
    }

    /// Diarizes a whole recording (16 kHz mono); identical to streaming it through `StreamingDiarizer`.
    public func run(_ audio: [Float]) throws -> (StreamingDiarizer.Output, StreamingDiarizer.Trace) {
        var steps = chunkSchedule(numSamples: audio.count, mode: mode, factor: factor).map { Step(spec: $0) }
        guard !steps.isEmpty else {
            return (StreamingDiarizer.Output(numSpeakers: numSpeakers), StreamingDiarizer.Trace())
        }

        // 1. front end + feature stacking of every chunk, in parallel
        let window = bundle.window, mel = bundle.melFilters
        let workers = min(steps.count, ProcessInfo.processInfo.activeProcessorCount)
        steps.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: workers) { w in
                let fe = FeatureExtractor(window: window, melFilters: mel)
                for k in stride(from: w, to: buf.count, by: workers) {
                    let spec = buf[k].spec
                    let feats = audio.withUnsafeBufferPointer {
                        fe.extract(UnsafeBufferPointer(rebasing: $0[spec.start..<spec.end]), center: spec.center,
                                   isLast: spec.isLast)
                    }
                    buf[k].numFrames = feats.count / FeatureExtractor.numMels
                    buf[k].embeds = embedder.embed(feats, numFrames: buf[k].numFrames)
                }
            }
        }

        // 2. cache / FIFO structure of every step (deterministic) and static device assignment
        let cacheConfig = bundle.config.speakerCache
        let lengths = SpeakerCache(config: cacheConfig, hidden: hidden, numSpeakers: numSpeakers,
                                   silenceEmbeds: bundle.silenceEmbeds)
        var nc = 0, nf = 0, epoch = 0, outRows = 0, leaves = 0
        for k in steps.indices {
            steps[k].numEmbeds = steps[k].embeds.count / hidden
            steps[k].numChunk = steps[k].numEmbeds - steps[k].spec.lookahead
            guard steps[k].numChunk >= 1 else { throw DiarizationError.invalidState("chunk without scored frames") }
            steps[k].firstID = steps[k].spec.melStart / factor
            steps[k].numCache = nc
            steps[k].numFifo = nf
            steps[k].epoch = epoch
            steps[k].outOffset = outRows
            outRows += min(steps[k].numChunk * factor, steps[k].numFrames)
            let popped = lengths.numPopped(nf + steps[k].numChunk)
            if popped > 0 {
                steps[k].isPop = true
                steps[k].onExact = exactRunner != nil
                nc = min(nc + popped, cacheConfig.speakerCacheLength)
                nf = nf + steps[k].numChunk - popped
                epoch += 1
            } else {
                nf += steps[k].numChunk
                steps[k].onCPU = cpuRunner != nil && cpuStride > 0 && leaves % cpuStride == cpuStride - 1
                leaves += 1
            }
        }

        // 3. embedding of every scored frame, as the FIFO holds it (from the step that scored it)
        let numIDs = steps.last!.firstID + steps.last!.numChunk
        var table = [Float](repeating: 0, count: numIDs * hidden)
        for step in steps {
            let rows = step.numChunk * hidden
            table.withUnsafeMutableBufferPointer { t in
                step.embeds.withUnsafeBufferPointer {
                    (t.baseAddress! + step.firstID * hidden).update(from: $0.baseAddress!, count: rows)
                }
            }
        }

        // 4. run: pop steps form a chain through the speaker-cache states ("epochs"), the rest fan out
        var epochs = [SpeakerCache.State?](repeating: nil, count: epoch + 1)
        epochs[0] = SpeakerCache.State()
        // written concurrently (disjoint rows per step): a raw buffer, not a Swift array
        let output = UnsafeMutablePointer<Float>.allocate(capacity: max(outRows * numSpeakers, 1))
        defer { output.deallocate() }
        var trace = StreamingDiarizer.Trace()
        trace.cacheIDs = Array(repeating: [], count: steps.count)
        trace.fifoIDs = Array(repeating: [], count: steps.count)
        trace.stepSeconds = Array(repeating: 0, count: steps.count)
        trace.stepLengths = Array(repeating: 0, count: steps.count)
        let gpuQueue = StepQueue(steps.indices.filter { !steps[$0].onCPU && !steps[$0].onExact })
        let cpuQueue = StepQueue(steps.indices.filter { steps[$0].onCPU })
        let exactQueue = StepQueue(steps.indices.filter { steps[$0].onExact })
        var failure: Error?
        let condition = NSCondition()

        // next ready step of a queue: pop steps (critical chain) first, then the earliest
        func take(_ queue: StepQueue) -> Int? {
            condition.lock()
            defer { condition.unlock() }
            while true {
                if failure != nil || queue.items.isEmpty { return nil }
                var best: Int?
                for (i, k) in queue.items.enumerated() where epochs[steps[k].epoch] != nil {
                    if steps[k].isPop { best = i; break }
                    if best == nil { best = i }
                }
                if let best { return queue.items.remove(at: best) }
                condition.wait()
            }
        }

        let silence = bundle.silenceEmbeds
        func process(_ k: Int, on runner: EncoderRunner) throws {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let step = steps[k]
            condition.lock()
            let cacheState = epochs[step.epoch]!
            condition.unlock()
            let fifoStart = step.firstID - step.numFifo
            let length = step.numCache + step.numFifo + step.numEmbeds
            let seq = runner.sequence
            cacheState.cacheEmbeds.withUnsafeBufferPointer {
                if $0.count > 0 { seq.update(from: $0.baseAddress!, count: $0.count) }
            }
            table.withUnsafeBufferPointer {
                (seq + step.numCache * hidden).update(from: $0.baseAddress! + fifoStart * hidden,
                                                      count: step.numFifo * hidden)
            }
            step.embeds.withUnsafeBufferPointer {
                (seq + (step.numCache + step.numFifo) * hidden).update(from: $0.baseAddress!, count: $0.count)
            }
            let logits = try runner.run(length: length)

            let offset = step.numCache + step.numFifo
            let rows = min(step.numChunk * factor, step.numFrames)
            (output + step.outOffset * numSpeakers).update(
                from: logits.baseAddress! + offset * factor * numSpeakers, count: rows * numSpeakers)
            var next: SpeakerCache.State?
            if step.isPop {
                let probs = pooledProbabilities(logits, length: length, factor: factor, numSpeakers: numSpeakers)
                let cache = SpeakerCache(config: cacheConfig, hidden: hidden, numSpeakers: numSpeakers,
                                         silenceEmbeds: silence)
                var state = cacheState
                state.fifoEmbeds = Array(table[(fifoStart * hidden)..<(step.firstID * hidden)])
                state.fifoIDs = Array(fifoStart..<step.firstID)
                cache.state = state
                cache.update(chunkEmbeds: step.embeds[0..<(step.numChunk * hidden)],
                             chunkIDs: Array(step.firstID..<(step.firstID + step.numChunk)), probs: probs)
                next = cache.state
                next!.fifoEmbeds = []
                next!.fifoIDs = []
            }
            condition.lock()
            trace.cacheIDs[k] = cacheState.cacheIDs
            trace.fifoIDs[k] = Array(fifoStart..<step.firstID)
            trace.stepSeconds[k] = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            trace.stepLengths[k] = length
            if let next {
                epochs[step.epoch + 1] = next
                condition.broadcast()
            }
            condition.unlock()
        }

        let group = DispatchGroup()
        func worker(_ runner: EncoderRunner, queue: StepQueue) {
            group.enter()
            let thread = Thread {
                defer { group.leave() }
                while true {
                    guard let k = take(queue) else { return }
                    do {
                        try process(k, on: runner)
                    } catch {
                        condition.lock()
                        failure = failure ?? error
                        condition.broadcast()
                        condition.unlock()
                        return
                    }
                }
            }
            thread.qualityOfService = .userInitiated
            thread.start()
        }
        for runner in gpuRunners { worker(runner, queue: gpuQueue) }
        if let cpuRunner { worker(cpuRunner, queue: cpuQueue) }
        if let exactRunner { worker(exactRunner, queue: exactQueue) }
        group.wait()
        if let failure { throw failure }
        let logits = Array(UnsafeBufferPointer(start: output, count: outRows * numSpeakers))
        return (StreamingDiarizer.Output(logits: logits, numFrames: outRows, numSpeakers: numSpeakers), trace)
    }
}

private final class StepQueue {
    var items: [Int]
    init(_ items: [Int]) { self.items = items }
}
