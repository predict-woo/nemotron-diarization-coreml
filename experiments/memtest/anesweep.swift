import CoreML
import Darwin
func mem() -> (Double, Double) {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e6, Double(info.resident_size) / 1e6)
}
let url = URL(filePath: CommandLine.arguments[1])
let lengths = CommandLine.arguments[2].split(separator: ",").map { Int($0)! }
var models: [MLModel] = []
for L in lengths {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine; cfg.functionName = "len_\(L)"
    let t0 = Date()
    let m = try! MLModel(contentsOf: url, configuration: cfg)
    let load = Date().timeIntervalSince(t0)
    let p = try! MLDictionaryFeatureProvider(dictionary: [
        "sequence": try! MLMultiArray(shape: [1, 512, 1, L as NSNumber], dataType: .float16),
        "key_bias": try! MLMultiArray(shape: [1, L as NSNumber, 1, 1], dataType: .float16),
        "valid": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float16)])
    _ = try! m.prediction(from: p); _ = try! m.prediction(from: p)
    var ts: [Double] = []
    for _ in 0..<15 { let t = Date(); _ = try! m.prediction(from: p); ts.append(Date().timeIntervalSince(t)) }
    models.append(m)
    let (f, r) = mem()
    print(String(format: "len_%d: %.1f ms (load %.1fs)  footprint %.0f MB  rss %.0f MB", L, ts.sorted()[7] * 1000, load, f, r))
}
