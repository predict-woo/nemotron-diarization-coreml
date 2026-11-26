// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "NemotronDiarization",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "NemotronDiarization", targets: ["NemotronDiarization"]),
        .executable(name: "nemotron-diarize", targets: ["nemotron-diarize"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        // PyTorch's CPU top-k (std::nth_element), whose tie-breaking the speaker cache must reproduce exactly
        .target(name: "CTorchTopK", path: "swift/Sources/CTorchTopK", publicHeadersPath: "include"),
        .target(
            name: "NemotronDiarization",
            dependencies: ["CTorchTopK"],
            path: "swift/Sources/NemotronDiarization",
            linkerSettings: [.linkedFramework("Accelerate"), .linkedFramework("CoreML")]
        ),
        .executableTarget(
            name: "nemotron-diarize",
            dependencies: ["NemotronDiarization", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "swift/Sources/nemotron-diarize",
            linkerSettings: [.linkedFramework("AVFoundation")]
        ),
    ],
    swiftLanguageModes: [.v5],
    cxxLanguageStandard: .cxx17
)
