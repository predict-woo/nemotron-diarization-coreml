import CoreML
import Foundation
let url = URL(filePath: CommandLine.arguments[1])
func load(_ units: MLComputeUnits) -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units; cfg.functionName = "len_576"
    return try! MLModel(contentsOf: url, configuration: cfg)
}
func feeds() -> MLFeatureProvider {
    let seq = try! MLMultiArray(shape: [1, 576, 512], dataType: .float32)
    let kb = try! MLMultiArray(shape: [1, 1, 1, 576], dataType: .float32)
    return try! MLDictionaryFeatureProvider(dictionary: ["sequence": seq, "key_bias": kb])
}
let gpu = load(.cpuAndGPU), cpu = load(.cpuOnly)
let fg = feeds(), fc = feeds()
_ = try! gpu.prediction(from: fg); _ = try! cpu.prediction(from: fc)
func alone(_ m: MLModel, _ f: MLFeatureProvider, _ n: Int) -> Double {
    let t = Date(); for _ in 0..<n { _ = try! m.prediction(from: f) }; return Date().timeIntervalSince(t) / Double(n) * 1000
}
print(String(format: "GPU alone %.1f ms/step, CPU alone %.1f ms/step", alone(gpu, fg, 20), alone(cpu, fc, 4)))
let duration = 8.0
var counts = [0, 0]
let lock = NSLock()
let group = DispatchGroup()
let stop = Date().addingTimeInterval(duration)
for (i, (m, f)) in [(gpu, fg), (cpu, fc)].enumerated() {
    group.enter()
    Thread {
        while Date() < stop { _ = try! m.prediction(from: f); lock.lock(); counts[i] += 1; lock.unlock() }
        group.leave()
    }.start()
}
group.wait()
print(String(format: "concurrent: GPU %d steps (%.1f ms each), CPU %d steps (%.1f ms each) -> %.1f ms/step combined",
             counts[0], duration / Double(counts[0]) * 1000, counts[1], duration / Double(max(counts[1], 1)) * 1000,
             duration / Double(counts[0] + counts[1]) * 1000))
