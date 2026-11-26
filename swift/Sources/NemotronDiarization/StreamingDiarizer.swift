import CoreML
import Foundation

/// Streaming speaker diarization with Nemotron-3-Diarization on CoreML.
///
/// Push 16 kHz mono audio in blocks of any size; every time a chunk and its look-ahead have arrived, one encoder
/// step runs and the chunk's 10 ms speaker logits are returned (`sigmoid` gives speaker activity probabilities,
/// speakers ordered by first arrival). `finish()` scores the frames left at the end of the stream. The chunking,
/// front end and speaker-cache updates reproduce the HF Transformers streaming forward step for step.
public final class StreamingDiarizer {
    public struct Output {
        /// Row-major (numFrames x numSpeakers) speaker logits, one row per 10 ms.
        public var logits: [Float]
        public var numFrames: Int
        public let numSpeakers: Int

        public var probabilities: [Float] { logits.map { 1 / (1 + expf(-$0)) } }

        public init(logits: [Float] = [], numFrames: Int = 0, numSpeakers: Int) {
            self.logits = logits
            self.numFrames = numFrames
            self.numSpeakers = numSpeakers
        }
    }

    /// Per-step record of the speaker-cache state (global encoder-frame ids, -1 = silence embedding), for tests.
    public struct Trace {
        public var cacheIDs: [[Int]] = []
        public var fifoIDs: [[Int]] = []
        public var stepSeconds: [Double] = []
        public var stepLengths: [Int] = []

        public init() {}
    }

    public let mode: StreamingMode
    public let numSpeakers: Int
    public private(set) var trace = Trace()
    public var recordTrace = true

    private let bundle: ModelBundle
    private let features: FeatureExtractor
    private let embedder: Embedder
    private let runner: EncoderRunner
    /// fp32 runner for the steps that update the speaker cache (Neural Engine bundles with an exact model)
    private let exactRunner: EncoderRunner?
    /// Steps that update the speaker cache run on the exact fp32 model when the bundle has one.
    public var exactCacheSteps = true
    private var cache: SpeakerCache
    private let hidden: Int
    private let factor: Int

    // stream position
    private var audio: [Float] = []  // samples from `audioOffset` on
    private var audioOffset = 0
    private var totalSamples = 0
    private var started = false
    private var finished = false
    private var melIndex = 0  // first mel frame of the next chunk

    /// `computeUnits` defaults to what the bundle's model is built for (Neural Engine or GPU).
    public init(bundleURL: URL, mode: StreamingMode = .lowLatency, computeUnits: MLComputeUnits? = nil) throws {
        bundle = try ModelBundle(url: bundleURL)
        let cfg = bundle.config
        self.mode = mode
        hidden = cfg.hiddenSize
        factor = cfg.subsamplingFactor
        numSpeakers = cfg.numSpeakers
        let needed = cfg.speakerCache.speakerCacheLength + cfg.speakerCache.fifoLength + mode.chunkLength
            + mode.rightContext
        guard needed <= cfg.maxSequenceLength else {
            throw DiarizationError.invalidBundle("model length \(cfg.maxSequenceLength) < longest step \(needed)")
        }
        features = FeatureExtractor(window: bundle.window, melFilters: bundle.melFilters)
        embedder = Embedder(weight: bundle.stackingWeight, hidden: hidden, factor: factor)
        runner = try EncoderRunner(models: try bundle.encoderModels(
            computeUnits: computeUnits ?? bundle.defaultComputeUnits))
        exactRunner = try bundle.exactEncoderModels().map { try EncoderRunner(models: $0) }
        cache = SpeakerCache(config: cfg.speakerCache, hidden: hidden, numSpeakers: numSpeakers,
                             silenceEmbeds: bundle.silenceEmbeds)
    }

    /// Clears the stream state (the model stays loaded).
    public func reset() {
        cache = SpeakerCache(config: bundle.config.speakerCache, hidden: hidden, numSpeakers: numSpeakers,
                             silenceEmbeds: bundle.silenceEmbeds)
        trace = Trace()
        audio = []
        audioOffset = 0
        totalSamples = 0
        started = false
        finished = false
        melIndex = 0
    }

    /// Runs every model bucket once so that real steps do not pay for GPU warm-up.
    public func warmUp() throws {
        try runner.warmUp()
        try exactRunner?.warmUp()
    }

    // MARK: schedule (identical to the model card's streaming generator, see Schedule.swift)

    private var samplesFirstChunk: Int { mode.samplesFirstChunk(factor: factor) }
    private var samplesPerChunk: Int { mode.samplesPerChunk(factor: factor) }
    private func chunkStart(_ melFrame: Int) -> Int { StreamingMode.chunkStart(melFrame) }

    /// Appends audio and runs every chunk that is complete.
    public func push(_ samples: UnsafeBufferPointer<Float>) throws -> Output {
        guard !finished else { throw DiarizationError.invalidState("push after finish") }
        audio.append(contentsOf: samples)
        totalSamples += samples.count
        var out = Output(logits: [], numFrames: 0, numSpeakers: numSpeakers)
        if !started && totalSamples >= samplesFirstChunk {
            try step(from: 0, to: samplesFirstChunk, center: true, isLast: false, lookahead: mode.rightContext,
                     melStart: 0, into: &out)
            started = true
            melIndex = mode.chunkLength * factor
        }
        while started && totalSamples >= chunkStart(melIndex) + samplesPerChunk {
            let start = chunkStart(melIndex)
            try step(from: start, to: start + samplesPerChunk, center: false, isLast: false,
                     lookahead: mode.rightContext, melStart: melIndex, into: &out)
            melIndex += mode.chunkLength * factor
        }
        if started {  // keep only what the next chunk needs
            let keepFrom = chunkStart(melIndex)
            if keepFrom > audioOffset {
                audio.removeFirst(keepFrom - audioOffset)
                audioOffset = keepFrom
            }
        }
        return out
    }

    public func push(_ samples: [Float]) throws -> Output {
        try samples.withUnsafeBufferPointer { try push($0) }
    }

    /// Ends the stream: scores every remaining frame (the last chunk has no look-ahead).
    public func finish() throws -> Output {
        guard !finished else { throw DiarizationError.invalidState("finish called twice") }
        finished = true
        var out = Output(logits: [], numFrames: 0, numSpeakers: numSpeakers)
        if !started {  // shorter than one chunk: a single centered chunk
            if totalSamples > 0 {
                try step(from: 0, to: totalSamples, center: true, isLast: false, lookahead: 0, melStart: 0, into: &out)
            }
        } else {
            try step(from: chunkStart(melIndex), to: totalSamples, center: false, isLast: true, lookahead: 0,
                     melStart: melIndex, into: &out)
        }
        return out
    }

    // MARK: one encoder step

    private func step(from start: Int, to end: Int, center: Bool, isLast: Bool, lookahead: Int, melStart: Int,
                      into out: inout Output) throws {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let feats = audio.withUnsafeBufferPointer {
            features.extract(UnsafeBufferPointer(rebasing: $0[(start - audioOffset)..<(end - audioOffset)]),
                             center: center, isLast: isLast)
        }
        let numFrames = feats.count / FeatureExtractor.numMels
        guard numFrames > 0 else { return }
        let embeds = embedder.embed(feats, numFrames: numFrames)
        let numEmbeds = embeds.count / hidden
        let numChunk = numEmbeds - lookahead
        guard numChunk >= 1 else { throw DiarizationError.invalidState("chunk without scored frames") }
        let firstID = melStart / factor

        if recordTrace {
            trace.cacheIDs.append(cache.cacheIDs)
            trace.fifoIDs.append(cache.fifoIDs)
        }
        let offset = cache.numCache + cache.numFifo
        let length = offset + numEmbeds
        let pops = cache.willPop(numChunkFrames: numChunk)
        let runner = (pops && exactCacheSteps ? exactRunner : nil) ?? self.runner
        let seq = runner.sequence
        cache.cacheEmbeds.withUnsafeBufferPointer { seq.update(from: $0.baseAddress!, count: $0.count) }
        cache.fifoEmbeds.withUnsafeBufferPointer {
            if $0.count > 0 { (seq + cache.numCache * hidden).update(from: $0.baseAddress!, count: $0.count) }
        }
        embeds.withUnsafeBufferPointer { (seq + offset * hidden).update(from: $0.baseAddress!, count: $0.count) }

        let logits = try runner.run(length: length)
        let probs = pops ? pooledProbabilities(logits, length: length, factor: factor, numSpeakers: numSpeakers) : nil
        cache.update(chunkEmbeds: embeds[0..<(numChunk * hidden)], chunkIDs: Array(firstID..<(firstID + numChunk)),
                     probs: probs)

        // the chunk's own frames, at 10 ms; the last chunk's feature-stacking padding is dropped
        let rows = min(numChunk * factor, numFrames)
        let s = numSpeakers
        out.logits.append(contentsOf: logits[(offset * factor * s)..<((offset * factor + rows) * s)])
        out.numFrames += rows
        if recordTrace {
            trace.stepSeconds.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            trace.stepLengths.append(length)
        }
    }
}
