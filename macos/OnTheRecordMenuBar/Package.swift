// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "OnTheRecordMenuBar",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "OnTheRecordMenuBar", targets: ["OnTheRecordMenuBar"]),
    ],
    targets: [
        .executableTarget(name: "OnTheRecordMenuBar"),
    ]
)
