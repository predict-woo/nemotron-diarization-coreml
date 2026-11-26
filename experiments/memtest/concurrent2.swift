import CoreML
import Foundation
let url = URL(filePath: CommandLine.arguments[1])
func load(_ units: MLComputeUnits, _ L: Int) -> MLModel {
    let cfg = MLModelConfiguration(); cfg.computeUnits = units; cfg.functionName = "len_\(L)"
    return try! MLModel(contentsOf: url, configuration: cfg)
}
func feeds(_ L: Int) -> MLFeatureProvider {
    try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)])
}
func alone(_ m: MLModel, _ f: MLFeatureProvider, _ n: Int) -> Double {
    _ = try! m.prediction(from: f)
    let t = Date(); for _ in 0..<n { _ = try! m.prediction(from: f) }; return Date().timeIntervalSince(t) / Double(n) * 1000
}
for L in [384, 448, 512, 576] { print(String(format: "CPU len_%d: %.1f ms", L, alone(load(.cpuOnly, L), feeds(L), 3))) }
func concurrent(_ workers: [(MLModel, MLFeatureProvider)], _ label: String) {
    let duration = 6.0
    var counts = [Int](repeating: 0, count: workers.count)
    let lock = NSLock(), group = DispatchGroup(), stop = Date().addingTimeInterval(duration)
    for (i, (m, f)) in workers.enumerated() {
        _ = try! m.prediction(from: f)
        group.enter()
        Thread { while Date() < stop { _ = try! m.prediction(from: f); lock.lock(); counts[i] += 1; lock.unlock() }; group.leave() }.start()
    }
    group.wait()
    print(String(format: "%@: %@ -> %.1f ms/step combined", label, counts.map(String.init).joined(separator: "+"), duration / Double(counts.reduce(0, +)) * 1000))
}
let g1 = load(.cpuAndGPU, 576), g2 = load(.cpuAndGPU, 576), c1 = load(.cpuOnly, 576)
concurrent([(g1, feeds(576)), (g2, feeds(576))], "2 GPU streams")
concurrent([(g1, feeds(576)), (g2, feeds(576)), (c1, feeds(576))], "2 GPU + CPU")
