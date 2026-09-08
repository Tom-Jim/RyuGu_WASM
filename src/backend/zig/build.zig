const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const wasm = target.result.cpu.arch == .wasm32;
    const cxx_flags: []const []const u8 = if (wasm) &.{ "-std=c++20", "-fno-exceptions" } else &.{"-std=c++20"};
    const module = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libcpp = true,
    });

    const bridge = if (wasm) b.addExecutable(.{
        .name = "ryugu_backend",
        .root_module = module,
    }) else b.addLibrary(.{
        .name = "ryugu_basilisk_bridge",
        .linkage = .static,
        .root_module = module,
    });
    if (wasm) {
        bridge.entry = .disabled;
        bridge.rdynamic = true;
        bridge.export_memory = true;
        module.export_symbol_names = &.{ "__wasm_call_ctors", "ryugu_alloc", "ryugu_free", "ryugu_basilisk_protocol_version", "ryugu_direct_sum_eval", "ryugu_radial_boost_eval", "ryugu_werner_eval", "ryugu_werner_reset_cache", "ryugu_exafmm_eval", "ryugu_flups_free_space_eval", "ryugu_scheduler_reset", "ryugu_scheduler_advance", "ryugu_scheduler_time" };
        module.addCMacro("RYUGU_BROWSER_SERIAL", "1");
        module.addCSourceFile(.{ .file = b.path("scheduler.cpp"), .flags = cxx_flags });
        inline for (.{ "_GeneralModuleFiles/sys_model.cpp", "_GeneralModuleFiles/sys_model_task.cpp", "system_model/sys_process.cpp", "system_model/sim_model.cpp", "utilities/moduleIdGenerator/moduleIdGenerator.cpp", "utilities/bskLogging.cpp" }) |file| {
            module.addCSourceFile(.{ .file = b.path("../../../C++/basilisk/src/architecture/" ++ file), .flags = cxx_flags });
        }
        module.addCMacro("BOOST_DISABLE_THREADS", "1");
        module.addCMacro("BOOST_MATH_DISABLE_THREADS", "1");
    }
    module.addCSourceFile(.{
        .file = b.path("basilisk_bridge.cpp"),
        .flags = cxx_flags,
    });
    module.addIncludePath(b.path("."));
    if (wasm) module.addCSourceFile(.{ .file = b.path("wasm_exports.cpp"), .flags = &.{"-std=c++20"} });
    {
        const root = b.option([]const u8, "basilisk-root", "Path to an AVSLab Basilisk checkout") orelse "../../../C++/basilisk";
        module.addIncludePath(.{ .cwd_relative = root });
        module.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ root, "src" }) });
        module.addCMacro("RYUGU_BASILISK_NATIVE", "1");
        module.addIncludePath(b.path("../../../C++/eigen"));
        module.addCSourceFile(.{
            .file = .{ .cwd_relative = b.pathJoin(&.{ root, "src/simulation/dynamics/gravityEffector/polyhedralGravityModel.cpp" }) },
            .flags = cxx_flags,
        });
    }
    if (b.option([]const u8, "boost-root", "Path to a Boost installation")) |root| {
        module.addIncludePath(.{ .cwd_relative = root });
        module.addCMacro("RYUGU_BOOST_NATIVE", "1");
    } else {
        inline for (.{ "math", "config", "assert", "core", "static_assert", "throw_exception", "type_traits", "predef" }) |name| {
            module.addIncludePath(b.path("../../../C++/boost/libs/" ++ name ++ "/include"));
        }
        module.addCMacro("RYUGU_BOOST_NATIVE", "1");
    }
    if (b.option([]const u8, "basilisk-lib", "Directory containing libbasilisk")) |path| {
        module.addLibraryPath(.{ .cwd_relative = path });
        module.linkSystemLibrary("basilisk", .{ .use_pkg_config = .no });
    }
    if (b.option([]const u8, "exafmm-root", "Path to an exafmm-t checkout")) |root| {
        module.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ root, "include" }) });
        module.addCMacro("RYUGU_EXAFMM_NATIVE", "1");
        if (wasm) {
            module.addIncludePath(b.path("browser"));
            module.addIncludePath(b.path("../../../C++/fftw3-release/api"));
            module.addObjectFile(b.path("deps/fftw/libfftw3.a"));
            module.addCSourceFile(.{ .file = b.path("browser/blas.cpp"), .flags = cxx_flags });
        } else {
            module.linkSystemLibrary("fftw3", .{ .use_pkg_config = .no });
            module.linkSystemLibrary("fftw3_omp", .{ .use_pkg_config = .no });
        }
    }
    if (b.option([]const u8, "flups-root", "Path to a FLUPS checkout")) |root| {
        module.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ root, "src" }) });
        module.addCMacro("RYUGU_FLUPS_NATIVE", "1");
        if (wasm) {
            module.addIncludePath(b.path("browser"));
            module.addIncludePath(b.path("../../../C++/fftw3-release/api"));
            module.addObjectFile(b.path("deps/flups/libflups.a"));
        }
    }
    if (b.option([]const u8, "flups-lib", "Directory containing libflups")) |path| {
        module.addLibraryPath(.{ .cwd_relative = path });
        module.linkSystemLibrary("flups", .{ .use_pkg_config = .no });
        module.linkSystemLibrary("mpi", .{ .use_pkg_config = .no });
        module.linkSystemLibrary("fftw3", .{ .use_pkg_config = .no });
    }

    b.installArtifact(bridge);
}
