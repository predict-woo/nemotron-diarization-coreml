import CoreML
let path = URL(filePath: CommandLine.arguments[1])
let n = Int(CommandLine.arguments[2])!
var models: [MLModel] = []
for L in [384, 448, 512, 576].prefix(n) {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU; cfg.functionName = "len_\(L)"
    let m = try! MLModel(contentsOf: path, configuration: cfg)
    let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)])
    _ = try! m.prediction(from: p)
    models.append(m)
}
print("ready"); fflush(stdout)
sleep(20)
