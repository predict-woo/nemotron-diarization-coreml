import CoreML
import Darwin
func mem() -> (Double, Double) {
    var info = task_vm_info_data_t(); var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    _ = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) } }
    return (Double(info.phys_footprint) / 1e6, Double(info.resident_size) / 1e6)
}
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
var hints = MLOptimizationHints(); hints.reshapeFrequency = .infrequent
cfg.optimizationHints = hints
let t0 = Date()
let m = try! MLModel(contentsOf: URL(filePath: CommandLine.arguments[1]), configuration: cfg)
print(String(format: "load %.1fs  footprint %.0f MB rss %.0f MB", Date().timeIntervalSince(t0), mem().0, mem().1))
let lengths = stride(from: 13, through: 541, by: 9).map { $0 } + [541]
var firsts: [Double] = []
for (i, n) in lengths.enumerated() {
    let seq = try! MLMultiArray(shape: [1, n as NSNumber, 512], dataType: .float32)
    let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": seq])
    let t = Date(); _ = try! m.prediction(from: p); firsts.append(Date().timeIntervalSince(t))
    if (i + 1) % 15 == 0 || i == lengths.count - 1 {
        print(String(format: "%3d shapes (L=%d): footprint %.0f MB rss %.0f MB, mean first-call %.0f ms", i + 1, n, mem().0, mem().1, firsts.suffix(15).reduce(0, +) / Double(min(15, firsts.count)) * 1000))
    }
}
// revisit shapes: cached?
var ts: [Double] = []
for n in [22, 301, 445, 526, 13, 535] {
    let p = try! MLDictionaryFeatureProvider(dictionary: ["sequence": try! MLMultiArray(shape: [1, n as NSNumber, 512], dataType: .float32)])
    let t = Date(); _ = try! m.prediction(from: p); ts.append(Date().timeIntervalSince(t) * 1000)
}
print("revisit:", ts.map { String(format: "%.0f", $0) }.joined(separator: " "), "ms")
