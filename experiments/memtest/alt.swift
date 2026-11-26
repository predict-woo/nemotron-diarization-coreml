import CoreML
let path = URL(filePath: CommandLine.arguments[1])
let buckets = [384, 448, 512, 576]
var models: [Int: MLModel] = [:], feeds: [Int: MLFeatureProvider] = [:]
let t0 = Date()
for L in buckets {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU; cfg.functionName = "len_\(L)"
    models[L] = try! MLModel(contentsOf: path, configuration: cfg)
    feeds[L] = try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)])
    _ = try! models[L]!.prediction(from: feeds[L]!)
}
print(String(format: "loaded + warmed %d functions in %.1fs", buckets.count, Date().timeIntervalSince(t0)))
// steady-state cycle: L = 322, 331, ..., then wraps (as the FIFO fills and pops)
var Ls: [Int] = []
var l = 322
for _ in 0..<200 { Ls.append(l); l += 9; if l > 541 { l -= 222 } }
var total = 0.0, per: [Int: [Double]] = [:]
for L in Ls {
    let b = buckets.first { $0 >= L }!
    let t = Date(); _ = try! models[b]!.prediction(from: feeds[b]!); let dt = Date().timeIntervalSince(t)
    total += dt; per[b, default: []].append(dt)
}
print(String(format: "cycle of %d steps: mean %.1f ms/step", Ls.count, total / Double(Ls.count) * 1000))
for b in buckets { let v = per[b]!.sorted(); print(String(format: "  bucket %d: n=%d median %.1f ms max %.1f", b, v.count, v[v.count / 2] * 1000, v.last! * 1000)) }
