const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Static library for C linkage
    const lib = b.addStaticLibrary(.{
        .name = "vagus",
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibC();
    b.installArtifact(lib);

    // Shared library
    const shared = b.addSharedLibrary(.{
        .name = "vagus",
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
    });
    shared.linkLibC();
    b.installArtifact(shared);

    // Unit tests (vagus.zig internal tests)
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
    });
    unit_tests.linkLibC();
    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Integration tests (vagus_test.zig)
    const integration_tests = b.addTest(.{
        .root_source_file = b.path("vagus_test.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.linkLibC();
    const run_integration_tests = b.addRunArtifact(integration_tests);

    // Test step runs both
    const test_step = b.step("test", "Run all vagus tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_integration_tests.step);

    // Header installation
    const install_header = b.addInstallHeaderFile(
        b.path("vagus.h"),
        "vagus.h",
    );
    b.getInstallStep().dependOn(&install_header.step);
}
