const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.createModule(.{
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const lib = b.addLibrary(.{ .name = "vagus", .root_module = mod, .linkage = .static });
    b.installArtifact(lib);

    const shared_mod = b.createModule(.{
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const shared = b.addLibrary(.{ .name = "vagus", .root_module = shared_mod, .linkage = .dynamic });
    b.installArtifact(shared);

    const unit_mod = b.createModule(.{
        .root_source_file = b.path("vagus.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const unit_tests = b.addTest(.{ .root_module = unit_mod });
    const run_unit = b.addRunArtifact(unit_tests);

    const integ_mod = b.createModule(.{
        .root_source_file = b.path("vagus_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const integ_tests = b.addTest(.{ .root_module = integ_mod });
    const run_integ = b.addRunArtifact(integ_tests);

    const test_step = b.step("test", "Run all vagus tests");
    test_step.dependOn(&run_unit.step);
    test_step.dependOn(&run_integ.step);

    const install_header = b.addInstallHeaderFile(b.path("vagus.h"), "vagus.h");
    b.getInstallStep().dependOn(&install_header.step);
}
