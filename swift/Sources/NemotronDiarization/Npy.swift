import Foundation

/// Minimal .npy writer (format 1.0, little-endian, C order) for handing runs to the Python comparison tools.
public enum Npy {
    public static func write(_ values: [Float], shape: [Int], to url: URL) throws {
        try write(values.withUnsafeBufferPointer { Data(buffer: $0) }, descr: "<f4", shape: shape, to: url)
    }

    public static func write(_ values: [Double], shape: [Int], to url: URL) throws {
        try write(values.withUnsafeBufferPointer { Data(buffer: $0) }, descr: "<f8", shape: shape, to: url)
    }

    public static func write(_ values: [Int32], shape: [Int], to url: URL) throws {
        try write(values.withUnsafeBufferPointer { Data(buffer: $0) }, descr: "<i4", shape: shape, to: url)
    }

    /// Rows of ids padded with `pad` to the longest row (at least one column).
    public static func writePadded(_ rows: [[Int]], pad: Int32 = -2, to url: URL) throws {
        let width = max(rows.map(\.count).max() ?? 0, 1)
        var flat = [Int32](repeating: pad, count: rows.count * width)
        for (r, row) in rows.enumerated() {
            for (c, v) in row.enumerated() { flat[r * width + c] = Int32(v) }
        }
        try write(flat, shape: [rows.count, width], to: url)
    }

    private static func write(_ payload: Data, descr: String, shape: [Int], to url: URL) throws {
        let shapeText = shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': \(shapeText), }"
        let unpadded = 10 + header.utf8.count + 1
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var data = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        let length = UInt16(header.utf8.count)
        data.append(contentsOf: [UInt8(length & 0xff), UInt8(length >> 8)])
        data.append(contentsOf: Array(header.utf8))
        data.append(payload)
        try data.write(to: url)
    }
}
