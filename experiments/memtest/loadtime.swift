import CoreML
import Foundation
let url = URL(filePath: CommandLine.arguments[1])
let parallel = CommandLine.arguments.count > 2
let buckets = [192, 384, 448, 512, 576]
func feeds(_ L: Int) -> MLFeatureProvider {
    try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)])
}
let t0 = Date()
if parallel {
    DispatchQueue.concurrentPerform(iterations: buckets.count) { i in
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU; cfg.functionName = "len_\(buckets[i])"
        let m = try! MLModel(contentsOf: url, configuration: cfg)
        _ = try! m.prediction(from: feeds(buckets[i]))
    }
    print(String(format: "parallel load + first prediction of %d functions: %.2fs", buckets.count, Date().timeIntervalSince(t0)))
} else {
    for L in buckets {
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU; cfg.functionName = "len_\(L)"
        let t1 = Date(); let m = try! MLModel(contentsOf: url, configuration: cfg); let load = Date().timeIntervalSince(t1)
        let t2 = Date(); _ = try! m.prediction(from: feeds(L)); let first = Date().timeIntervalSince(t2)
        print(String(format: "len_%d: load %.2fs, first prediction %.2fs", L, load, first))
    }
    print(String(format: "sequential total %.2fs", Date().timeIntervalSince(t0)))
}
