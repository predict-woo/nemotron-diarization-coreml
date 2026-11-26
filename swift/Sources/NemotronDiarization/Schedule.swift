import Foundation

/// Chunking of a stream, identical to the model card's streaming generator (HF processor).
extension StreamingMode {
    /// Samples the first chunk needs (centered analysis windows).
    func samplesFirstChunk(factor: Int) -> Int {
        ((chunkLength + rightContext) * factor - 1) * FeatureExtractor.hop + FeatureExtractor.winLength / 2
    }

    /// Samples every later chunk spans (uncentered windows), from `chunkStart` of its first mel frame.
    func samplesPerChunk(factor: Int) -> Int {
        (chunkLength + rightContext) * factor * FeatureExtractor.hop + FeatureExtractor.winLength
    }

    /// First audio sample of the chunk whose first mel frame is `melFrame`.
    static func chunkStart(_ melFrame: Int) -> Int { melFrame * FeatureExtractor.hop - FeatureExtractor.nFFT / 2 }
}

/// One encoder step of a stream.
struct ChunkSpec {
    let start: Int  // audio samples [start, end)
    let end: Int
    let center: Bool  // first chunk: centered analysis windows
    let isLast: Bool  // last uncentered chunk: the front end pads the end of the audio
    let lookahead: Int  // trailing look-ahead encoder frames, not scored
    let melStart: Int  // global index of the chunk's first mel frame
}

/// Every step of a stream of `numSamples` samples, in order.
func chunkSchedule(numSamples: Int, mode: StreamingMode, factor: Int) -> [ChunkSpec] {
    let first = mode.samplesFirstChunk(factor: factor)
    guard numSamples >= first else {  // shorter than one chunk: a single centered chunk, every frame scored
        return numSamples > 0 ? [ChunkSpec(start: 0, end: numSamples, center: true, isLast: false, lookahead: 0,
                                           melStart: 0)] : []
    }
    var chunks = [ChunkSpec(start: 0, end: first, center: true, isLast: false, lookahead: mode.rightContext,
                            melStart: 0)]
    var mel = mode.chunkLength * factor
    let span = mode.samplesPerChunk(factor: factor)
    while StreamingMode.chunkStart(mel) + span <= numSamples {
        let start = StreamingMode.chunkStart(mel)
        chunks.append(ChunkSpec(start: start, end: start + span, center: false, isLast: false,
                                lookahead: mode.rightContext, melStart: mel))
        mel += mode.chunkLength * factor
    }
    chunks.append(ChunkSpec(start: StreamingMode.chunkStart(mel), end: numSamples, center: false, isLast: true,
                            lookahead: 0, melStart: mel))
    return chunks
}

/// Encoder-rate probabilities of every step position: sigmoid averaged over each group of `factor` frames
/// (sequential sum then division, the order of the original's avg_pool1d).
func pooledProbabilities(_ logits: UnsafeBufferPointer<Float>, length: Int, factor: Int, numSpeakers s: Int)
    -> [Float]
{
    var probs = [Float](repeating: 0, count: length * s)
    for i in 0..<length {
        for spk in 0..<s {
            var total: Float = 0
            for j in 0..<factor {
                let p = 1 / (1 + expf(-logits[(i * factor + j) * s + spk]))
                total = j == 0 ? p : total + p
            }
            probs[i * s + spk] = total / Float(factor)
        }
    }
    return probs
}
