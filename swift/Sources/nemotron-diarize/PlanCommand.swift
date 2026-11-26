import ArgumentParser
import CoreML
import Foundation

struct PlanCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "plan",
        abstract: "Show where CoreML runs the operations of every model function."
    )

    @OptionGroup var model: ModelOptions

    func run() throws {
        let url = try model.bundleURL()
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: url.appending(path: "config.json")))
            as? [String: Any] ?? [:]
        let layout = config["model_layout"] as? String ?? "fixed"
        let buckets = (config["buckets"] as? [Int]) ?? [config["max_sequence_length"] as? Int ?? 0]
        let units = model.units?.computeUnits ?? (layout == "ane" ? .cpuAndNeuralEngine : .cpuAndGPU)
        let devices = ["Neural Engine", "GPU", "CPU"]
        print(Terminal.bold(pad("function", 12) + devices.map { lpad($0, 15) }.joined()))
        for length in buckets {
            let counts = try waitFor { try await operationCounts(url, length: length, layout: layout, units: units) }
            print(pad("len_\(length)", 12) + devices.map { lpad(String(counts[$0, default: 0]), 15) }.joined())
            for (device, n) in counts where !devices.contains(device) {
                print(Terminal.dim("  \(n) operations without a device (\(device))"))
            }
        }
    }
}

/// Operation counts per preferred compute device, for one function of the encoder.
private func operationCounts(_ bundle: URL, length: Int, layout: String, units: MLComputeUnits) async throws
    -> [String: Int]
{
    let configuration = MLModelConfiguration()
    configuration.computeUnits = units
    if layout == "buckets" || layout == "ane" { configuration.functionName = "len_\(length)" }
    let plan = try await MLComputePlan.load(contentsOf: bundle.appending(path: "Encoder.mlmodelc"),
                                            configuration: configuration)
    guard case .program(let program) = plan.modelStructure,
          let function = program.functions[configuration.functionName ?? "main"] else { return [:] }
    var counts: [String: Int] = [:]
    for op in function.block.operations where op.operatorName != "const" {
        switch plan.deviceUsage(for: op)?.preferred {
        case .neuralEngine: counts["Neural Engine", default: 0] += 1
        case .gpu: counts["GPU", default: 0] += 1
        case .cpu: counts["CPU", default: 0] += 1
        default: counts["none", default: 0] += 1
        }
    }
    return counts
}

/// Runs async work from a synchronous command.
private func waitFor<T>(_ work: @escaping () async throws -> T) throws -> T {
    let done = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var result: Result<T, Error>!
    Task {
        do { result = .success(try await work()) } catch { result = .failure(error) }
        done.signal()
    }
    done.wait()
    return try result.get()
}
