import Accelerate
import Foundation
func bench(_ m: Int, _ n: Int, _ k: Int, _ iters: Int = 30) -> Double {
    let a = [Float](repeating: 0.01, count: m * k), b = [Float](repeating: 0.01, count: k * n)
    var c = [Float](repeating: 0, count: m * n)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(m), Int32(n), Int32(k), 1, a, Int32(k), b, Int32(k), 0, &c, Int32(n))
    let t = Date()
    for _ in 0..<iters { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(m), Int32(n), Int32(k), 1, a, Int32(k), b, Int32(k), 0, &c, Int32(n)) }
    let dt = Date().timeIntervalSince(t) / Double(iters)
    return 2.0 * Double(m * n * k) / dt / 1e12
}
for (m, n, k, name) in [(541, 1536, 512, "QKV"), (541, 512, 512, "O"), (541, 2048, 512, "FC1"), (541, 512, 2048, "FC2"), (541, 541, 64, "QK^T head")] {
    print(String(format: "%-10@ %4dx%4dx%4d: %.2f TFLOPS", name as NSString, m, n, k, bench(m, n, k)))
}
