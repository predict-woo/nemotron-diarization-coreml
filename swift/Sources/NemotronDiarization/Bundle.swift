import CoreML
import Foundation

/// Streaming modes of the model card: (chunk length, look-ahead) in 80 ms encoder frames.
public enum StreamingMode: String, CaseIterable, Sendable {
    case lowLatency = "low_latency"  // 1.04 s
    case veryLowLatency = "very_low_latency"  // 0.64 s
    case ultraLowLatency = "ultra_low_latency"  // 0.32 s

    public var chunkLength: Int {
        switch self {
        case .lowLatency: 9
        case .veryLowLatency: 6
        case .ultraLowLatency: 3
        }
    }

    public var rightContext: Int {
        switch self {
        case .lowLatency: 4
        case .veryLowLatency: 2
        case .ultraLowLatency: 1
        }
    }
}

public struct SpeakerCacheConfig: Decodable, Sendable {
    public var fifoLength = 264
    public var speakerCacheUpdatePeriod = 222
    public var speakerCacheLength = 264
    public var silenceFramesPerSpeaker = 1
    public var predictionScoreThreshold = 0.25
    public var latestFramesScoreBoost = 0.05
    public var strongBoostRate = 0.75
    public var weakBoostRate = 1.5
    public var minPositiveScoresRate = 0.5

    enum CodingKeys: String, CodingKey {
        case fifoLength = "fifo_length"
        case speakerCacheUpdatePeriod = "speaker_cache_update_period"
        case speakerCacheLength = "speaker_cache_length"
        case silenceFramesPerSpeaker = "silence_frames_per_speaker"
        case predictionScoreThreshold = "prediction_score_threshold"
        case latestFramesScoreBoost = "latest_frames_score_boost"
        case strongBoostRate = "strong_boost_rate"
        case weakBoostRate = "weak_boost_rate"
        case minPositiveScoresRate = "min_positive_scores_rate"
    }
}

struct ExactModelConfig: Decodable {
    var modelLayout: String
    var buckets: [Int]
    var computeUnits: String

    enum CodingKeys: String, CodingKey {
        case modelLayout = "model_layout"
        case buckets
        case computeUnits = "compute_units"
    }
}

struct BundleConfig: Decodable {
    var modelLayout: String
    var buckets: [Int]?
    var exact: ExactModelConfig?
    var maxSequenceLength: Int
    var hiddenSize: Int
    var numSpeakers: Int
    var subsamplingFactor: Int
    var speakerCache: SpeakerCacheConfig

    enum CodingKeys: String, CodingKey {
        case modelLayout = "model_layout"
        case buckets
        case exact
        case maxSequenceLength = "max_sequence_length"
        case hiddenSize = "hidden_size"
        case numSpeakers = "num_speakers"
        case subsamplingFactor = "subsampling_factor"
        case speakerCache = "speaker_cache"
    }
}

/// The model bundle written by scripts/build_bundle.py.
struct ModelBundle {
    let url: URL
    let config: BundleConfig
    let window: [Float]
    let melFilters: [Float]  // 128 x 257
    let stackingWeight: [Float]  // 512 x 1024
    let silenceEmbeds: [Float]  // 512

    init(url: URL) throws {
        self.url = url
        config = try JSONDecoder().decode(BundleConfig.self, from: Data(contentsOf: url.appending(path: "config.json")))
        window = try Self.floats(url.appending(path: "window.f32"), count: 400)
        melFilters = try Self.floats(url.appending(path: "mel_filters.f32"), count: 128 * 257)
        stackingWeight = try Self.floats(url.appending(path: "stacking_weight.f32"), count: config.hiddenSize * 128 * 8)
        silenceEmbeds = try Self.floats(url.appending(path: "silence_embeds.f32"), count: config.hiddenSize)
    }

    var encoderURL: URL { url.appending(path: "Encoder.mlmodelc") }

    /// Loads the encoder for one compute device; `buckets` restricts the bucket lengths loaded (memory).
    /// fp32 model for the steps that update the speaker cache, if the bundle has one (Neural Engine bundles):
    /// with it, every speaker-cache decision is made on fp32 probabilities, as in the original.
    func exactEncoderModels() throws -> EncoderModels? {
        guard let exact = config.exact else { return nil }
        return try EncoderModels(
            url: url.appending(path: "EncoderExact.mlmodelc"),
            computeUnits: exact.computeUnits == "gpu" ? .cpuAndGPU : .cpuOnly, layout: exact.modelLayout,
            buckets: exact.buckets, hidden: config.hiddenSize, numSpeakers: config.numSpeakers,
            upsample: config.subsamplingFactor)
    }

    /// The compute units the bundle's model is built for: the Neural Engine formulation, or the GPU.
    var defaultComputeUnits: MLComputeUnits { config.modelLayout == "ane" ? .cpuAndNeuralEngine : .cpuAndGPU }

    func encoderModels(computeUnits: MLComputeUnits, buckets: [Int]? = nil) throws -> EncoderModels {
        var lengths = config.buckets ?? [config.maxSequenceLength]
        if let buckets { lengths = lengths.filter { buckets.contains($0) || $0 == lengths.last } }
        return try EncoderModels(url: encoderURL, computeUnits: computeUnits, layout: config.modelLayout,
                                 buckets: lengths, hidden: config.hiddenSize, numSpeakers: config.numSpeakers,
                                 upsample: config.subsamplingFactor)
    }

    static func floats(_ url: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == count * MemoryLayout<Float>.size else {
            throw DiarizationError.invalidBundle("\(url.lastPathComponent): \(data.count) bytes, expected \(count) floats")
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

public enum DiarizationError: Error, CustomStringConvertible {
    case invalidBundle(String)
    case invalidState(String)
    case model(String)

    public var description: String {
        switch self {
        case .invalidBundle(let m): "invalid model bundle: \(m)"
        case .invalidState(let m): "invalid state: \(m)"
        case .model(let m): "model error: \(m)"
        }
    }
}
