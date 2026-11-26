import CoreML
import Foundation
let url = URL(filePath: CommandLine.arguments[1])
func run(_ label: String, _ configure: (MLModelConfiguration) -> Void) {
    for L in [448, 576] {
        let cfg = MLModelConfiguration(); cfg.functionName = "len_\(L)"; configure(cfg)
        let t0 = Date()
        let m = try! MLModel(contentsOf: url, configuration: cfg)
        let load = Date().timeIntervalSince(t0)
        let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)])
        _ = try! m.prediction(from: p); _ = try! m.prediction(from: p)
        var ts: [Double] = []
        for _ in 0..<15 { let t = Date(); _ = try! m.prediction(from: p); ts.append(Date().timeIntervalSince(t)) }
        print(String(format: "%-34@ len_%d: %.1f ms (load %.1fs)", label as NSString, L, ts.sorted()[7] * 1000, load))
    }
}
run("cpuAndGPU") { $0.computeUnits = .cpuAndGPU }
run("all") { $0.computeUnits = .all }
run("cpuAndGPU + fastPrediction") { cfg in
    cfg.computeUnits = .cpuAndGPU
    var h = MLOptimizationHints(); h.specializationStrategy = .fastPrediction; cfg.optimizationHints = h
}
