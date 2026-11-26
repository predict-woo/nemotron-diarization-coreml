import CoreML
import Darwin
func footprintMB() -> Double {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return Double(info.phys_footprint) / 1e6
}
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
var models: [MLModel] = []
print(String(format: "start footprint %.0f MB", footprintMB()))
for path in CommandLine.arguments.dropFirst() {
    let m = try! MLModel(contentsOf: URL(filePath: path), configuration: cfg)
    let L = (m.modelDescription.inputDescriptionsByName["sequence"]!.multiArrayConstraint!.shape[1]).intValue
    let seq = try! MLMultiArray(shape: [1, L as NSNumber, 512], dataType: .float32)
    let kb = try! MLMultiArray(shape: [1, 1, 1, L as NSNumber], dataType: .float32)
    let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": seq, "key_bias": kb])
    _ = try! m.prediction(from: p); _ = try! m.prediction(from: p)
    models.append(m)
    print(String(format: "after %@: footprint %.0f MB", (path as NSString).lastPathComponent, footprintMB()))
}
