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

    // Tests
    const tests = b.addTest(.{
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run vagus tests");
    test_step.dependOn(&run_tests.step);

    // Header installation
    const install_header = b.addInstallHeaderFile(
        b.path("vagus.h"),
        "vagus.h",
    );
    b.getInstallStep().dependOn(&install_header.step);
}
