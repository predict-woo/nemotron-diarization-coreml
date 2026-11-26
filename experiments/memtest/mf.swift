import CoreML
import Darwin
func footprintMB() -> Double {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return Double(info.phys_footprint) / 1e6
}
func rssMB() -> Double {
    var info = mach_task_basic_info(); var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count) } }
    return Double(info.resident_size) / 1e6
}
let path = URL(filePath: CommandLine.arguments[1])
var models: [MLModel] = []
let t0 = Date()
for L in [384, 448, 512, 576] {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU; cfg.functionName = "len_\(L)"
    let m = try! MLModel(contentsOf: path, configuration: cfg)
    let seq = try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32)
    let kb = try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)
    let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": seq, "key_bias": kb])
    _ = try! m.prediction(from: p); _ = try! m.prediction(from: p)
    var ts: [Double] = []
    for _ in 0..<10 { let t = Date(); _ = try! m.prediction(from: p); ts.append(Date().timeIntervalSince(t)) }
    models.append(m)
    print(String(format: "len_%d: %.1f ms   footprint %.0f MB  rss %.0f MB  (t=%.1fs)", L, ts.sorted()[5] * 1000, footprintMB(), rssMB(), Date().timeIntervalSince(t0)))
}
