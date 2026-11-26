import CTorchTopK
import Foundation

/// Arrival-Order Speaker Cache + FIFO queue of the streaming model, a port of `Nemotron3DiarizationSpeakerCache`
/// (batch size 1) in float32. Every slot carries the global encoder-frame id it holds (-1: the learned silence
/// embedding) so runs can be compared state by state with the original.
final class SpeakerCache {
    static let silenceID = -1

    let config: SpeakerCacheConfig
    let hidden: Int
    let numSpeakers: Int
    let silenceEmbeds: [Float]
    let minPositiveScores: Int
    let numStrongBoosted: Int
    let numWeakBoosted: Int

    private(set) var cacheEmbeds: [Float] = []  // numCache x hidden
    private(set) var cacheProbs: [Float] = []  // numCache x numSpeakers
    private(set) var cacheIDs: [Int] = []
    private(set) var fifoEmbeds: [Float] = []  // numFifo x hidden
    private(set) var fifoIDs: [Int] = []
    private(set) var isCompressed = false

    /// Complete cache + FIFO contents, to resume the cache from any step.
    struct State {
        var cacheEmbeds: [Float] = []
        var cacheProbs: [Float] = []
        var cacheIDs: [Int] = []
        var fifoEmbeds: [Float] = []
        var fifoIDs: [Int] = []
        var isCompressed = false
    }

    var state: State {
        get {
            State(cacheEmbeds: cacheEmbeds, cacheProbs: cacheProbs, cacheIDs: cacheIDs, fifoEmbeds: fifoEmbeds,
                  fifoIDs: fifoIDs, isCompressed: isCompressed)
        }
        set {
            cacheEmbeds = newValue.cacheEmbeds
            cacheProbs = newValue.cacheProbs
            cacheIDs = newValue.cacheIDs
            fifoEmbeds = newValue.fifoEmbeds
            fifoIDs = newValue.fifoIDs
            isCompressed = newValue.isCompressed
        }
    }

    var numCache: Int { cacheIDs.count }
    var numFifo: Int { fifoIDs.count }

    init(config: SpeakerCacheConfig, hidden: Int, numSpeakers: Int, silenceEmbeds: [Float]) {
        self.config = config
        self.hidden = hidden
        self.numSpeakers = numSpeakers
        self.silenceEmbeds = silenceEmbeds
        let budget = config.speakerCacheLength / numSpeakers - config.silenceFramesPerSpeaker
        minPositiveScores = Int((Double(budget) * config.minPositiveScoresRate).rounded(.down))
        numStrongBoosted = Int((Double(budget) * config.strongBoostRate).rounded(.down))
        numWeakBoosted = Int((Double(budget) * config.weakBoostRate).rounded(.down))
    }

    /// Whether pushing `numChunkFrames` frames moves frames to the speaker cache (and so needs probabilities).
    func willPop(numChunkFrames: Int) -> Bool { numFifo + numChunkFrames > config.fifoLength }

    func numPopped(_ numFifoFrames: Int) -> Int {
        guard numFifoFrames > config.fifoLength else { return 0 }
        return min(max(config.speakerCacheUpdatePeriod, numFifoFrames - config.fifoLength), numFifoFrames)
    }

    /// Pushes the chunk's scored frames. `probs` (encoder rate, row-major L x numSpeakers) cover the whole step
    /// input [cache, FIFO, chunk + look-ahead]; they are only read when frames move to the speaker cache.
    func update(chunkEmbeds: ArraySlice<Float>, chunkIDs: [Int], probs: [Float]?) {
        let nc = numCache, nf = numFifo, n = chunkIDs.count
        var newFifoEmbeds = fifoEmbeds + chunkEmbeds
        var newFifoIDs = fifoIDs + chunkIDs
        let popped = numPopped(newFifoIDs.count)
        if popped > 0 {
            guard let probs else { preconditionFailure("probabilities are required when the FIFO overflows") }
            let s = numSpeakers
            let fifoProbs = probs[(nc * s)..<((nc + nf + n) * s)]  // no look-back frames: fifo and chunk adjacent
            var probsOut = isCompressed ? cacheProbs : Array(probs[0..<(nc * s)])
            probsOut += fifoProbs.prefix(popped * s)
            var embedsOut = cacheEmbeds + newFifoEmbeds.prefix(popped * hidden)
            var idsOut = cacheIDs + newFifoIDs.prefix(popped)
            newFifoEmbeds.removeFirst(popped * hidden)
            newFifoIDs.removeFirst(popped)
            if idsOut.count > config.speakerCacheLength {
                let rows = compressionIndices(probs: probsOut, numFrames: idsOut.count)
                let numFrames = idsOut.count
                var e = [Float](), p = [Float](), ids = [Int]()
                e.reserveCapacity(rows.count * hidden)
                p.reserveCapacity(rows.count * s)
                for r in rows {
                    if r == numFrames {
                        e += silenceEmbeds
                        p += [Float](repeating: 0, count: s)
                        ids.append(Self.silenceID)
                    } else {
                        e += embedsOut[(r * hidden)..<((r + 1) * hidden)]
                        p += probsOut[(r * s)..<((r + 1) * s)]
                        ids.append(idsOut[r])
                    }
                }
                embedsOut = e
                probsOut = p
                idsOut = ids
                isCompressed = true
            }
            cacheEmbeds = embedsOut
            cacheProbs = probsOut
            cacheIDs = idsOut
        }
        fifoEmbeds = newFifoEmbeds
        fifoIDs = newFifoIDs
    }

    // MARK: compression

    /// Rows of [frames..., silence] kept by the compressed cache, in cache order.
    func compressionIndices(probs: [Float], numFrames: Int) -> [Int] {
        let s = numSpeakers
        var scores = frameScores(probs: probs, numFrames: numFrames)
        let latest = Float(config.latestFramesScoreBoost)
        if numFrames > config.speakerCacheLength {
            for i in (config.speakerCacheLength * s)..<(numFrames * s) { scores[i] += latest }
        }
        boost(&scores, numFrames: numFrames, count: numStrongBoosted, by: Float(-2.0 * log(0.5)))
        boost(&scores, numFrames: numFrames, count: numWeakBoosted, by: Float(-log(0.5)))

        // speaker-major flattening of [frames + silence rows] x speakers; silence rows score +inf
        let numSil = config.silenceFramesPerSpeaker
        let numScored = numFrames + numSil
        var flat = [Float](repeating: .infinity, count: numScored * s)
        for spk in 0..<s {
            for t in 0..<numFrames { flat[spk * numScored + t] = scores[t * s + spk] }
        }
        let k = config.speakerCacheLength
        var idx = topK(flat, k)
        let sentinel = numScored * s
        for i in 0..<k where flat[idx[i]] == -.infinity { idx[i] = sentinel }
        idx.sort()
        return idx.map { $0 == sentinel ? numFrames : min($0 % numScored, numFrames) }
    }

    func frameScores(probs: [Float], numFrames: Int) -> [Float] {
        let s = numSpeakers
        let thr = Float(config.predictionScoreThreshold)
        let logHalf = Float(log(0.5))
        var scores = [Float](repeating: 0, count: numFrames * s)
        var logProbs = [Float](repeating: 0, count: s), logComp = [Float](repeating: 0, count: s)
        for t in 0..<numFrames {
            for j in 0..<s {
                let p = probs[t * s + j]
                logProbs[j] = logf(max(p, thr))
                logComp[j] = logf(max(1 - p, thr))
            }
            var sumComp = logComp[0]  // sequential, the summation order of torch.sum
            for j in 1..<s { sumComp += logComp[j] }
            for j in 0..<s {
                let p = probs[t * s + j]
                let score = logProbs[j] - logComp[j] + sumComp - logHalf
                scores[t * s + j] = p > 0.5 ? score : -.infinity
            }
        }
        // a speaker with enough positive (single-speaker) frames drops its non-positive (overlapped) speech frames
        for j in 0..<s {
            var positives = 0
            for t in 0..<numFrames where scores[t * s + j] > 0 { positives += 1 }
            guard positives >= minPositiveScores else { continue }
            for t in 0..<numFrames {
                let i = t * s + j
                if !(scores[i] > 0) && probs[i] > 0.5 { scores[i] = -.infinity }
            }
        }
        return scores
    }

    /// Adds `value` to the `count` highest scores of every speaker.
    private func boost(_ scores: inout [Float], numFrames: Int, count: Int, by value: Float) {
        let s = numSpeakers
        var column = [Float](repeating: 0, count: numFrames)
        for j in 0..<s {
            for t in 0..<numFrames { column[t] = scores[t * s + j] }
            for t in topK(column, count) { scores[t * s + j] += value }
        }
    }

    private func topK(_ values: [Float], _ k: Int) -> [Int] {
        var out = [Int64](repeating: 0, count: k)
        values.withUnsafeBufferPointer { torch_topk_largest($0.baseAddress!, Int64(values.count), Int64(k), &out) }
        return out.map { Int($0) }
    }
}
