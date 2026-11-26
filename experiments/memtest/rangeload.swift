import CoreML
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
let t0 = Date()
let m = try! MLModel(contentsOf: URL(filePath: CommandLine.arguments[1]), configuration: cfg)
print(String(format: "load %.1fs", Date().timeIntervalSince(t0)))
for n in [322, 430, 541] {
    let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, n as NSNumber, 512], dataType: .float32)])
    _ = try! m.prediction(from: p)
    let t = Date(); for _ in 0..<5 { _ = try! m.prediction(from: p) }
    print(String(format: "L=%d: %.1f ms", n, Date().timeIntervalSince(t) / 5 * 1000))
}
