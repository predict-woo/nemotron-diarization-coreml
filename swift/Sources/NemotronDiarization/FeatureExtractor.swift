import Accelerate

/// Log-mel front end of one streaming chunk, matching `NemotronAsrStreamingFeatureExtractor` + the HF processor:
/// preemphasis 0.97 -> (last uncentered chunk: 96 zero samples) -> STFT (n_fft 512, hop 160, symmetric Hann 400
/// centered in the frame; first chunk `center`ed with 256 zeros on both sides) -> |X|^2 -> 128 slaney mel filters
/// -> log(x + 2^-24). Only the frames the processor keeps are returned.
final class FeatureExtractor {
    static let nFFT = 512
    static let hop = 160
    static let winLength = 400
    static let numMels = 128
    static let numBins = nFFT / 2 + 1
    static let logGuard = Float(pow(2.0, -24.0))
    static let preemphasis: Float = 0.97

    private let frameWindow: [Float]
    private let melFiltersT: [Float]  // numBins x numMels, row-major (the filter bank, transposed)
    private let dft: vDSP_DFT_Setup
    private var frame = [Float](repeating: 0, count: nFFT)
    private var evenIn = [Float](repeating: 0, count: nFFT / 2)
    private var oddIn = [Float](repeating: 0, count: nFFT / 2)
    private var realOut = [Float](repeating: 0, count: nFFT / 2)
    private var imagOut = [Float](repeating: 0, count: nFFT / 2)

    init(window: [Float], melFilters: [Float]) {
        precondition(window.count == Self.winLength && melFilters.count == Self.numMels * Self.numBins)
        var w = [Float](repeating: 0, count: Self.nFFT)
        let left = (Self.nFFT - Self.winLength) / 2  // torch.stft centers the window inside the frame
        for i in 0..<Self.winLength { w[left + i] = window[i] }
        frameWindow = w
        var transposed = [Float](repeating: 0, count: melFilters.count)
        vDSP_mtrans(melFilters, 1, &transposed, 1, vDSP_Length(Self.numBins), vDSP_Length(Self.numMels))
        melFiltersT = transposed
        dft = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD)!
    }

    deinit { vDSP_DFT_DestroySetup(dft) }

    /// Number of frames `extract` returns for `numSamples` samples.
    static func numFrames(numSamples: Int, center: Bool, isLast: Bool) -> Int {
        let length = numSamples + (isLast && !center ? nFFT / 2 - hop : 0)
        if center { return length / hop }
        return length >= nFFT ? (length - nFFT) / hop + 1 : 0
    }

    /// Features of one chunk, row-major (numFrames x 128).
    func extract(_ audio: UnsafeBufferPointer<Float>, center: Bool, isLast: Bool) -> [Float] {
        let n = audio.count
        let numFrames = Self.numFrames(numSamples: n, center: center, isLast: isLast)
        guard numFrames > 0 else { return [] }
        let padLeft = center ? Self.nFFT / 2 : 0
        let tail = (isLast && !center ? Self.nFFT / 2 - Self.hop : 0) + (center ? Self.nFFT / 2 : 0)
        var signal = [Float](repeating: 0, count: padLeft + n + tail)
        signal[padLeft] = audio[0]
        for i in 1..<n { signal[padLeft + i] = audio[i] - Self.preemphasis * audio[i - 1] }

        var power = [Float](repeating: 0, count: numFrames * Self.numBins)
        signal.withUnsafeBufferPointer { sig in
            power.withUnsafeMutableBufferPointer { pow in
                for t in 0..<numFrames {
                    vDSP_vmul(sig.baseAddress! + t * Self.hop, 1, frameWindow, 1, &frame, 1, vDSP_Length(Self.nFFT))
                    for i in 0..<(Self.nFFT / 2) {
                        evenIn[i] = frame[2 * i]
                        oddIn[i] = frame[2 * i + 1]
                    }
                    vDSP_DFT_Execute(dft, evenIn, oddIn, &realOut, &imagOut)
                    // zrop packs DC in real[0] and Nyquist in imag[0], and is scaled by 2 (halving is exact)
                    let row = pow.baseAddress! + t * Self.numBins
                    let dc = 0.5 * realOut[0], nyquist = 0.5 * imagOut[0]
                    row[0] = Self.squaredMagnitude(dc, 0)
                    row[Self.nFFT / 2] = Self.squaredMagnitude(nyquist, 0)
                    for k in 1..<(Self.nFFT / 2) {
                        row[k] = Self.squaredMagnitude(0.5 * realOut[k], 0.5 * imagOut[k])
                    }
                }
            }
        }

        var mel = [Float](repeating: 0, count: numFrames * Self.numMels)
        // (numFrames x numBins) power @ (numBins x numMels) filters
        vDSP_mmul(power, 1, melFiltersT, 1, &mel, 1, vDSP_Length(numFrames), vDSP_Length(Self.numMels),
                  vDSP_Length(Self.numBins))
        for i in 0..<mel.count { mel[i] = logf(mel[i] + Self.logGuard) }
        return mel
    }

    /// The original takes the magnitude, then squares it.
    @inline(__always) static func squaredMagnitude(_ re: Float, _ im: Float) -> Float {
        let magnitude = (re * re + im * im).squareRoot()
        return magnitude * magnitude
    }
}

/// Feature stacking (8 consecutive mel frames, zero-padded) followed by the bias-free 1024 -> 512 projection.
final class Embedder {
    let weightT: [Float]  // (128 * 8) x hidden, row-major (the projection, transposed)
    let hidden: Int
    let factor: Int

    init(weight: [Float], hidden: Int, factor: Int) {
        var transposed = [Float](repeating: 0, count: weight.count)
        vDSP_mtrans(weight, 1, &transposed, 1, vDSP_Length(FeatureExtractor.numMels * factor), vDSP_Length(hidden))
        weightT = transposed
        self.hidden = hidden
        self.factor = factor
    }

    /// (numFrames x 128) features -> (ceil(numFrames / 8) x hidden) embeddings.
    func embed(_ features: [Float], numFrames: Int) -> [Float] {
        let numEmbeds = (numFrames + factor - 1) / factor
        let inDim = FeatureExtractor.numMels * factor
        var stacked = [Float](repeating: 0, count: numEmbeds * inDim)  // zero tail = padding of the last group
        stacked.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: features, count: features.count) }
        var out = [Float](repeating: 0, count: numEmbeds * hidden)
        // (numEmbeds x inDim) stacked features @ (inDim x hidden) projection
        vDSP_mmul(stacked, 1, weightT, 1, &out, 1, vDSP_Length(numEmbeds), vDSP_Length(hidden), vDSP_Length(inDim))
        return out
    }
}
