import CoreML
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
func pageAligned(_ count: Int) -> UnsafeMutablePointer<Float> {
    var p: UnsafeMutableRawPointer?
    posix_memalign(&p, 16384, (count * 4 + 16383) / 16384 * 16384)
    let f = p!.bindMemory(to: Float.self, capacity: count); f.initialize(repeating: 0.001, count: count); return f
}
func bench(_ m: MLModel, _ p: MLFeatureProvider, _ name: String) {
    _ = try! m.prediction(from: p); _ = try! m.prediction(from: p)
    var ts: [Double] = []
    for _ in 0..<10 { let t = Date(); _ = try! m.prediction(from: p); ts.append(Date().timeIntervalSince(t)) }
    print(name, String(format: "%.1f ms", ts.sorted()[5] * 1000))
}
let x = try! MLMultiArray(shape: [1, 541, 512], dataType: .float32)
let mc = try! MLModel(contentsOf: URL(filePath: "build/exp_mlp_const.mlmodelc"), configuration: cfg)
bench(mc, try! MLDictionaryFeatureProvider(dictionary: ["x": x]), "constants          ")
let mi = try! MLModel(contentsOf: URL(filePath: "build/exp_mlp_input.mlmodelc"), configuration: cfg)
let n1 = 31 * 512 * 2048
let w1 = try! MLMultiArray(dataPointer: pageAligned(n1), shape: [31, 512, 2048], dataType: .float32, strides: [NSNumber(value: 512 * 2048), 2048, 1])
let w2 = try! MLMultiArray(dataPointer: pageAligned(n1), shape: [31, 2048, 512], dataType: .float32, strides: [NSNumber(value: 512 * 2048), 512, 1])
bench(mi, try! MLDictionaryFeatureProvider(dictionary: ["x": x, "w1": w1, "w2": w2]), "inputs, page-aligned")
