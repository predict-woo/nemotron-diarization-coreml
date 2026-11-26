import CoreML
import Darwin
func mem() -> (Double, Double) {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e6, Double(info.resident_size) / 1e6)
}
let url = URL(filePath: CommandLine.arguments[1])
var models: [MLModel] = []
for L in [384, 448, 512, 576] {
    let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuOnly; cfg.functionName = "len_\(L)"
    let m = try! MLModel(contentsOf: url, configuration: cfg)
    _ = try! m.prediction(from: try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32), "key_bias": try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)]))
    models.append(m)
    let (f, r) = mem(); print(String(format: "CPU len_%d loaded: footprint %.0f MB, rss %.0f MB", L, f, r))
}
