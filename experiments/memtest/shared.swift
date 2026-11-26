import CoreML
import Foundation
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU; cfg.functionName = "len_576"
let m = try! MLModel(contentsOf: URL(filePath: CommandLine.arguments[1]), configuration: cfg)
func feeds() -> MLFeatureProvider {
    try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, 576, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, 576], dataType: .float32)])
}
_ = try! m.prediction(from: feeds())
for threads in [1, 2, 3] {
    let duration = 5.0
    var counts = [Int](repeating: 0, count: threads)
    let lock = NSLock(), group = DispatchGroup(), stop = Date().addingTimeInterval(duration)
    for i in 0..<threads {
        let f = feeds()
        group.enter()
        Thread { while Date() < stop { _ = try! m.prediction(from: f); lock.lock(); counts[i] += 1; lock.unlock() }; group.leave() }.start()
    }
    group.wait()
    print(String(format: "%d thread(s) on one MLModel: %.1f ms/step", threads, duration / Double(counts.reduce(0, +)) * 1000))
}
