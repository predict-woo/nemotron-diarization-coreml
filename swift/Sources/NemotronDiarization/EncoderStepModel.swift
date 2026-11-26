import Accelerate
import CoreML
import Foundation

/// The CoreML encoder step: [speaker cache, FIFO, chunk + look-ahead] (n x 512) -> speaker logits (8n x 8).
///
/// Layouts (bundle `model_layout`):
///   buckets - multifunction model, function `len_<L>` per bucket: `sequence` (1, L, 512) zero-padded to L +
///             `key_bias` (1, 1, 1, L) key mask; each step runs the smallest bucket that holds it
///   fixed   - one such function-less model
///   range   - `sequence` (1, n, 512) of the exact length (flexible shape)
///   ane     - Neural Engine formulation (float16, channels first), function `len_<L>` per bucket:
///             `sequence` (1, 512, 1, L), `key_bias` (1, L, 1, 1), `valid` (1, 1, 1, L) -> `logits` (1, 8, 8, L)
///
/// `EncoderModels` holds the loaded models of one compute device; they are shared by any number of
/// `EncoderRunner`s, each with its own buffers, which may predict concurrently.
final class EncoderModels {
    let layout: String
    let bucketLengths: [Int]
    let hidden: Int
    let numSpeakers: Int
    let upsample: Int
    fileprivate var models: [Int: MLModel] = [:]  // by bucket length (range: one model keyed by its maximum)

    var maxLength: Int { bucketLengths.last! }

    init(url: URL, computeUnits: MLComputeUnits, layout: String, buckets: [Int], hidden: Int, numSpeakers: Int,
         upsample: Int) throws {
        guard ["buckets", "fixed", "range", "ane"].contains(layout) else {
            throw DiarizationError.invalidBundle("unknown model layout \(layout)")
        }
        self.layout = layout
        bucketLengths = buckets.sorted()
        self.hidden = hidden
        self.numSpeakers = numSpeakers
        self.upsample = upsample
        // loading mostly waits on the compiler service: the buckets load ~10x faster in parallel
        let lengths = layout == "range" ? [bucketLengths.last!] : bucketLengths
        var loaded = [Result<MLModel, Error>?](repeating: nil, count: lengths.count)
        loaded.withUnsafeMutableBufferPointer { results in
            DispatchQueue.concurrentPerform(iterations: lengths.count) { i in
                let configuration = MLModelConfiguration()
                configuration.computeUnits = computeUnits
                if layout == "buckets" || layout == "ane" { configuration.functionName = "len_\(lengths[i])" }
                results[i] = Result { try MLModel(contentsOf: url, configuration: configuration) }
            }
        }
        for (length, result) in zip(lengths, loaded) { models[length] = try result!.get() }
    }

    /// First prediction of every bucket, concurrently (each pays a one-time GPU specialisation of ~0.25 s).
    func warmUp() throws {
        let lengths = Array(models.keys).sorted()
        let maskedLayout = layout != "range"
        var errors = [Error?](repeating: nil, count: lengths.count)
        errors.withUnsafeMutableBufferPointer { errs in
            DispatchQueue.concurrentPerform(iterations: lengths.count) { i in
                do {
                    let n = lengths[i]
                    var inputs: [String: MLFeatureValue]
                    if layout == "ane" {
                        inputs = [
                            "sequence": MLFeatureValue(multiArray: try MLMultiArray(
                                shape: [1, hidden, 1, n] as [NSNumber], dataType: .float16)),
                            "key_bias": MLFeatureValue(multiArray: try MLMultiArray(
                                shape: [1, n, 1, 1] as [NSNumber], dataType: .float16)),
                            "valid": MLFeatureValue(multiArray: try MLMultiArray(
                                shape: [1, 1, 1, n] as [NSNumber], dataType: .float16)),
                        ]
                    } else {
                        inputs = ["sequence": MLFeatureValue(multiArray: try MLMultiArray(
                            shape: [1, n, hidden] as [NSNumber], dataType: .float32))]
                        if maskedLayout {
                            inputs["key_bias"] = MLFeatureValue(multiArray: try MLMultiArray(
                                shape: [1, 1, 1, n] as [NSNumber], dataType: .float32))
                        }
                    }
                    _ = try models[n]!.prediction(from: try MLDictionaryFeatureProvider(dictionary: inputs))
                } catch {
                    errs[i] = error
                }
            }
        }
        if let error = errors.compactMap({ $0 }).first { throw error }
    }

    /// The bucket a step of `n` positions runs in.
    func bucket(for n: Int) -> Int? { bucketLengths.first { $0 >= n } }
}

/// Per-worker input/output buffers for `EncoderModels` (no per-step allocation).
final class EncoderRunner {
    private final class Bucket {
        let length: Int
        let model: MLModel
        let sequence: MLMultiArray
        let keyBias: MLMultiArray
        let logits: MLMultiArray
        var maskedFrom = -1  // valid length the key bias currently encodes

        init(length: Int, model: MLModel, sequence: MLMultiArray, keyBias: MLMultiArray, logits: MLMultiArray) {
            self.length = length
            self.model = model
            self.sequence = sequence
            self.keyBias = keyBias
            self.logits = logits
        }
    }

    /// Neural Engine bucket: float16 channel-first inputs, (1, 8, 8, L) float16 output.
    private final class ANEBucket {
        let length: Int
        let model: MLModel
        let sequence: MLMultiArray  // (1, hidden, 1, L) float16
        let keyBias: MLMultiArray  // (1, L, 1, 1)
        let valid: MLMultiArray  // (1, 1, 1, L)
        let logits: MLMultiArray  // (1, speakers, upsample, L)
        var maskedFrom = -1

        init(length: Int, model: MLModel, hidden: Int, numSpeakers: Int, upsample: Int) throws {
            self.length = length
            self.model = model
            sequence = try MLMultiArray(shape: [1, hidden, 1, length] as [NSNumber], dataType: .float16)
            keyBias = try MLMultiArray(shape: [1, length, 1, 1] as [NSNumber], dataType: .float16)
            valid = try MLMultiArray(shape: [1, 1, 1, length] as [NSNumber], dataType: .float16)
            logits = try MLMultiArray(shape: [1, numSpeakers, upsample, length] as [NSNumber], dataType: .float16)
        }
    }

    let models: EncoderModels
    private var aneBuckets: [ANEBucket] = []
    private var transposed: [Float] = []  // hidden x L staging for the ANE input
    private let sequenceStorage: UnsafeMutablePointer<Float>
    private let logitsStorage: UnsafeMutablePointer<Float>
    private var buckets: [Bucket] = []
    static let negativeBias: Float = -1e4

    var maxLength: Int { models.maxLength }

    init(models: EncoderModels) throws {
        self.models = models
        let maxLength = models.maxLength, hidden = models.hidden
        let outRow = models.upsample * models.numSpeakers
        sequenceStorage = .allocate(capacity: maxLength * hidden)
        sequenceStorage.initialize(repeating: 0, count: maxLength * hidden)
        logitsStorage = .allocate(capacity: maxLength * outRow)
        logitsStorage.initialize(repeating: 0, count: maxLength * outRow)
        guard models.layout != "range" else { return }
        if models.layout == "ane" {
            transposed = [Float](repeating: 0, count: hidden * maxLength)
            for length in models.bucketLengths {
                aneBuckets.append(try ANEBucket(length: length, model: models.models[length]!, hidden: hidden,
                                                numSpeakers: models.numSpeakers, upsample: models.upsample))
            }
            return
        }
        for length in models.bucketLengths {
            let sequence = try MLMultiArray(
                dataPointer: sequenceStorage, shape: [1, length, hidden] as [NSNumber], dataType: .float32,
                strides: [length * hidden, hidden, 1] as [NSNumber])
            let keyBias = try MLMultiArray(shape: [1, 1, 1, length] as [NSNumber], dataType: .float32)
            let logits = try MLMultiArray(
                dataPointer: logitsStorage, shape: [1, length * models.upsample, models.numSpeakers] as [NSNumber],
                dataType: .float32, strides: [length * outRow, models.numSpeakers, 1] as [NSNumber])
            buckets.append(Bucket(length: length, model: models.models[length]!, sequence: sequence, keyBias: keyBias,
                                  logits: logits))
        }
    }

    deinit {
        sequenceStorage.deallocate()
        logitsStorage.deallocate()
    }

    /// Row-major (maxLength x hidden) input buffer; the caller writes the first n rows before `run(length:)`.
    var sequence: UnsafeMutablePointer<Float> { sequenceStorage }

    /// Warms every bucket up, then runs each once through this runner's own buffers.
    func warmUp() throws {
        try models.warmUp()
        for b in (models.layout == "range" ? [models.maxLength] : models.bucketLengths) { _ = try run(length: b) }
    }

    /// Runs the step on the first `n` rows of `sequence`; returns the (8n x numSpeakers) logits.
    func run(length n: Int) throws -> UnsafeBufferPointer<Float> {
        guard n > 0 && n <= maxLength else {
            throw DiarizationError.invalidState("step length \(n) outside 1...\(maxLength)")
        }
        let hidden = models.hidden, numSpeakers = models.numSpeakers
        let outCount = n * models.upsample * numSpeakers
        if models.layout == "range" {
            let sequence = try MLMultiArray(
                dataPointer: sequenceStorage, shape: [1, n, hidden] as [NSNumber], dataType: .float32,
                strides: [n * hidden, hidden, 1] as [NSNumber])
            // a flexible-shape output cannot be backed by a caller buffer: copy it out
            let result = try models.models[models.maxLength]!.prediction(
                from: try MLDictionaryFeatureProvider(dictionary: ["sequence": MLFeatureValue(multiArray: sequence)]))
            guard let logits = result.featureValue(for: "logits")?.multiArrayValue, logits.count == outCount else {
                throw DiarizationError.model("unexpected logits output")
            }
            logits.withUnsafeBufferPointer(ofType: Float.self) { src in
                precondition(logits.strides.map(\.intValue) == [outCount, numSpeakers, 1], "non-contiguous logits")
                logitsStorage.update(from: src.baseAddress!, count: outCount)
            }
            return UnsafeBufferPointer(start: logitsStorage, count: outCount)
        }

        if models.layout == "ane" { return try runANE(length: n) }
        let bucket = buckets.first { $0.length >= n }!
        // padding rows must hold finite values (they are masked out); zero them
        (sequenceStorage + n * hidden).update(repeating: 0, count: (bucket.length - n) * hidden)
        if bucket.maskedFrom != n {
            bucket.keyBias.withUnsafeMutableBufferPointer(ofType: Float.self) { bias, _ in
                for i in 0..<bucket.length { bias[i] = i < n ? 0 : Self.negativeBias }
            }
            bucket.maskedFrom = n
        }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "sequence": MLFeatureValue(multiArray: bucket.sequence),
            "key_bias": MLFeatureValue(multiArray: bucket.keyBias),
        ])
        let options = MLPredictionOptions()
        options.outputBackings = ["logits": bucket.logits]
        _ = try bucket.model.prediction(from: inputs, options: options)
        return UnsafeBufferPointer(start: logitsStorage, count: outCount)
    }

    /// ANE step: rows -> channel-first float16 (padding zeroed), mask, predict, output back to (8n x speakers).
    private func runANE(length n: Int) throws -> UnsafeBufferPointer<Float> {
        let hidden = models.hidden, s = models.numSpeakers, k = models.upsample
        let bucket = aneBuckets.first { $0.length >= n }!
        let L = bucket.length
        // (n x hidden) fp32 rows -> (hidden x L) fp32, zero padded, -> fp16
        transposed.withUnsafeMutableBufferPointer { t in
            vDSP_mtrans(sequenceStorage, 1, t.baseAddress!, 1, vDSP_Length(hidden), vDSP_Length(n))
            if n < L {  // mtrans wrote a dense (hidden x n) block: spread rows to stride L, back to front
                for c in stride(from: hidden - 1, through: 1, by: -1) {
                    let src = t.baseAddress! + c * n, dst = t.baseAddress! + c * L
                    memmove(dst, src, n * MemoryLayout<Float>.size)
                    (t.baseAddress! + c * L + n).update(repeating: 0, count: L - n)
                }
                (t.baseAddress! + n).update(repeating: 0, count: L - n)
            }
            bucket.sequence.withUnsafeMutableBytes { raw, _ in
                var src = vImage_Buffer(data: t.baseAddress!, height: 1, width: vImagePixelCount(hidden * L),
                                        rowBytes: hidden * L * 4)
                var dst = vImage_Buffer(data: raw.baseAddress!, height: 1, width: vImagePixelCount(hidden * L),
                                        rowBytes: hidden * L * 2)
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
            }
        }
        if bucket.maskedFrom != n {
            bucket.keyBias.withUnsafeMutableBufferPointer(ofType: Float16.self) { bias, _ in
                for i in 0..<L { bias[i] = i < n ? 0 : Float16(Self.negativeBias) }
            }
            bucket.valid.withUnsafeMutableBufferPointer(ofType: Float16.self) { valid, _ in
                for i in 0..<L { valid[i] = i < n ? 1 : 0 }
            }
            bucket.maskedFrom = n
        }
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "sequence": MLFeatureValue(multiArray: bucket.sequence),
            "key_bias": MLFeatureValue(multiArray: bucket.keyBias),
            "valid": MLFeatureValue(multiArray: bucket.valid),
        ])
        let options = MLPredictionOptions()
        options.outputBackings = ["logits": bucket.logits]
        _ = try bucket.model.prediction(from: inputs, options: options)
        // (speakers, k, L) fp16 -> (n * k, speakers) fp32
        bucket.logits.withUnsafeBufferPointer(ofType: Float16.self) { out in
            for spk in 0..<s {
                for j in 0..<k {
                    let row = out.baseAddress! + (spk * k + j) * L
                    for i in 0..<n { logitsStorage[(i * k + j) * s + spk] = Float(row[i]) }
                }
            }
        }
        return UnsafeBufferPointer(start: logitsStorage, count: n * k * s)
    }
}
