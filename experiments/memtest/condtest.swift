import CoreML
import Darwin
func mem() -> (Double, Double) {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e6, Double(info.resident_size) / 1e6)
}
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
let t0 = Date()
let m = try! MLModel(contentsOf: URL(filePath: CommandLine.arguments[1]), configuration: cfg)
print(String(format: "load %.1fs  footprint %.0f MB rss %.0f MB", Date().timeIntervalSince(t0), mem().0, mem().1))
let x = try! MLMultiArray(shape: [1, 576, 512], dataType: .float32)
for i in 0..<x.count { x[i] = NSNumber(value: Float(i % 97) * 0.001) }
func feeds(_ b: Int) -> MLFeatureProvider {
    let sel = try! MLMultiArray(shape: [1], dataType: .int32); sel[0] = NSNumber(value: b)
    return try! MLDictionaryFeatureProvider(dictionary: ["x": x, "bucket": sel])
}
for b in 0..<4 {
    let f = feeds(b)
    let t = Date(); _ = try! m.prediction(from: f); let first = Date().timeIntervalSince(t)
    var ts: [Double] = []
    for _ in 0..<10 { let t = Date(); _ = try! m.prediction(from: f); ts.append(Date().timeIntervalSince(t)) }
    print(String(format: "branch %d: %.1f ms (first %.0f ms)  footprint %.0f MB rss %.0f MB", b, ts.sorted()[5] * 1000, first * 1000, mem().0, mem().1))
}
var ts: [Double] = []
for i in 0..<12 { let f = feeds(i % 4); let t = Date(); _ = try! m.prediction(from: f); ts.append(Date().timeIntervalSince(t) * 1000) }
print("alternating branches:", ts.map { String(format: "%.0f", $0) }.joined(separator: " "), "ms")
