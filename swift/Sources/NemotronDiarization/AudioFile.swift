import AVFoundation

public enum AudioFile {
    /// Reads an audio file as 16 kHz mono float32. Other rates / channel counts are converted with
    /// AVAudioConverter (the equivalence tests use files that are already 16 kHz mono, so no resampling happens).
    public static func load16kMono(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
              let chunk = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1 << 16)
        else { throw DiarizationError.invalidState("cannot allocate audio buffer") }
        // a single read may return fewer frames than requested: read until the end of the file
        while file.framePosition < file.length {
            try file.read(into: chunk)
            if chunk.frameLength == 0 { break }
            for c in 0..<Int(format.channelCount) {
                (buffer.floatChannelData![c] + Int(buffer.frameLength))
                    .update(from: chunk.floatChannelData![c], count: Int(chunk.frameLength))
            }
            buffer.frameLength += chunk.frameLength
        }
        guard buffer.frameLength == frames else {
            throw DiarizationError.invalidState("read \(buffer.frameLength) of \(frames) frames from \(url.path)")
        }
        if format.sampleRate == 16000 && format.channelCount == 1 && format.commonFormat == .pcmFormatFloat32 {
            return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        }
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: format, to: target) else {
            throw DiarizationError.invalidState("unsupported audio format \(format)")
        }
        let capacity = AVAudioFrameCount(Double(frames) * 16000 / format.sampleRate) + 1024
        let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)!
        var consumed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if consumed {
                status.pointee = .endOfStream
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        if let error { throw error }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }
}
