#include "basilisk_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(RYUGU_BOOST_NATIVE)
#include <boost/math/quadrature/gauss_kronrod.hpp>
#endif

#if defined(RYUGU_BASILISK_NATIVE)
#include "simulation/dynamics/gravityEffector/polyhedralGravityModel.h"
#endif

#if defined(RYUGU_EXAFMM_NATIVE)
#include "build_list.h"
#include "build_tree.h"
#include "laplace.h"
#endif

#if defined(RYUGU_FLUPS_NATIVE)
#include "flups.h"
#include <mpi.h>
#endif

namespace {
constexpr double kGravityConstant = 6.67430e-11;
constexpr double kPi = 3.141592653589793238462643383279502884;
#if defined(RYUGU_BASILISK_NATIVE)
std::unique_ptr<PolyhedralGravityModel> cached_polyhedron;
#endif

int direct_sum(const double *source_xyz, const double *source_mass,
              uint64_t source_count, const double position[3],
              double acceleration[3], double *potential) {
    if (source_xyz == nullptr || source_mass == nullptr || position == nullptr
        || acceleration == nullptr || potential == nullptr || source_count == 0) {
        return -2;
    }
    acceleration[0] = acceleration[1] = acceleration[2] = 0.0;
    *potential = 0.0;
    for (uint64_t index = 0; index < source_count; ++index) {
        const double dx = source_xyz[3 * index + 0] - position[0];
        const double dy = source_xyz[3 * index + 1] - position[1];
        const double dz = source_xyz[3 * index + 2] - position[2];
        const double radius_squared = std::max(dx * dx + dy * dy + dz * dz, 1.0e-24);
        const double inverse_radius = 1.0 / std::sqrt(radius_squared);
        const double scale = kGravityConstant * source_mass[index]
            * inverse_radius / radius_squared;
        acceleration[0] += scale * dx;
        acceleration[1] += scale * dy;
        acceleration[2] += scale * dz;
        *potential += kGravityConstant * source_mass[index] * inverse_radius;
    }
    return 0;
}
}

// Browser numerical ABI. Basilisk task scheduling is implemented in scheduler.cpp.
uint16_t ryugu_basilisk_protocol_version(void) {
    return 1;
}

// Legacy native-only placeholder, superseded by ryugu_scheduler_advance.
// int32_t ryugu_basilisk_step(const RyuguBasiliskState *state,
//                             const RyuguBasiliskForce *force,
//                             uint64_t step_ns) {
//     if (state == nullptr || force == nullptr || step_ns == 0) return -2;
//     return -1;
// }

int32_t ryugu_direct_sum_eval(const double *source_xyz,
                              const double *source_mass,
                              uint64_t source_count,
                              const double position_m[3],
                              double acceleration_mps2[3],
                              double *potential_m2ps2) {
    return direct_sum(source_xyz, source_mass, source_count, position_m,
                      acceleration_mps2, potential_m2ps2);
}

int32_t ryugu_radial_boost_eval(const double *cells,
                                uint64_t cell_count,
                                const double position[3],
                                double acceleration[3],
                                double *potential,
                                double abs_tolerance,
                                double rel_tolerance) {
#if defined(RYUGU_BOOST_NATIVE)
    if (cells == nullptr || position == nullptr || acceleration == nullptr
        || potential == nullptr || cell_count == 0) {
        return -2;
    }
    acceleration[0] = acceleration[1] = acceleration[2] = 0.0;
    *potential = 0.0;
    using Quadrature = boost::math::quadrature::gauss_kronrod<double, 61>;
    constexpr unsigned max_depth = 12;
    const double abs_tol = std::max(abs_tolerance, 1.0e-18);
    const double rel_tol = std::max(rel_tolerance, 1.0e-12);
    for (uint64_t index = 0; index < cell_count; ++index) {
        // Record layout: direction.xyz, solid_angle, r_inner, r_outer,
        // density, padding. The integrand is continuous inside each shell.
        const double *cell = cells + 8 * index;
        const double direction[3] = {cell[0], cell[1], cell[2]};
        const double solid_angle = cell[3];
        const double inner = cell[4];
        const double outer = std::max(cell[5], inner);
        const double density = cell[6];
        for (int component = 0; component < 3; ++component) {
            const auto integrand = [&](double radius) {
                const double sx = direction[0] * radius - position[0];
                const double sy = direction[1] * radius - position[1];
                const double sz = direction[2] * radius - position[2];
                const double radius_squared = std::max(sx * sx + sy * sy + sz * sz, 1.0e-24);
                const double volume_weight = solid_angle * radius * radius;
                const double displacement = component == 0 ? sx : (component == 1 ? sy : sz);
                return kGravityConstant * density * volume_weight * displacement
                    / (radius_squared * std::sqrt(radius_squared));
            };
            double error = 0.0;
            const double value = Quadrature::integrate(
                integrand, inner, outer, max_depth, rel_tol, &error);
            if (error > abs_tol + rel_tol * std::abs(value)) return -5;
            acceleration[component] += value;
        }
        const auto potential_integrand = [&](double radius) {
            const double sx = direction[0] * radius - position[0];
            const double sy = direction[1] * radius - position[1];
            const double sz = direction[2] * radius - position[2];
            const double distance = std::sqrt(std::max(sx * sx + sy * sy + sz * sz, 1.0e-24));
            return kGravityConstant * density * solid_angle * radius * radius / distance;
        };
        double error = 0.0;
        const double value = Quadrature::integrate(
            potential_integrand, inner, outer, max_depth, rel_tol, &error);
        if (error > abs_tol + rel_tol * std::abs(value)) return -5;
        *potential += value;
    }
    return 0;
#else
    (void)cells; (void)cell_count; (void)position; (void)acceleration;
    (void)potential; (void)abs_tolerance; (void)rel_tolerance;
    return -1;
#endif
}

int32_t ryugu_werner_eval(const double *vertices_xyz,
                          uint64_t vertex_count,
                          const uint32_t *facets,
                          uint64_t facet_count,
                          double mu_body,
                          const double position[3],
                          double acceleration[3],
                          double *potential) {
#if defined(RYUGU_BASILISK_NATIVE)
    if (vertices_xyz == nullptr || facets == nullptr || position == nullptr
        || acceleration == nullptr || potential == nullptr || vertex_count == 0 || facet_count == 0) {
        return -2;
    }
    if (!cached_polyhedron) {
        cached_polyhedron = std::make_unique<PolyhedralGravityModel>();
        cached_polyhedron->muBody = mu_body;
        cached_polyhedron->xyzVertex.resize(static_cast<Eigen::Index>(vertex_count), 3);
        cached_polyhedron->orderFacet.resize(static_cast<Eigen::Index>(facet_count), 3);
        for (uint64_t index = 0; index < vertex_count; ++index) {
            for (int component = 0; component < 3; ++component) {
                cached_polyhedron->xyzVertex(static_cast<Eigen::Index>(index), component) =
                    vertices_xyz[3 * index + component];
            }
        }
        for (uint64_t index = 0; index < facet_count; ++index) {
            for (int component = 0; component < 3; ++component) {
                cached_polyhedron->orderFacet(static_cast<Eigen::Index>(index), component) =
                    static_cast<int>(facets[3 * index + component] + 1);
            }
        }
        if (cached_polyhedron->initializeParameters().has_value()) {
            cached_polyhedron.reset();
            return -3;
        }
    }
    const Eigen::Vector3d position_eigen(position[0], position[1], position[2]);
    const Eigen::Vector3d result = cached_polyhedron->computeField(position_eigen);
    *potential = cached_polyhedron->computePotentialEnergy(position_eigen);
    acceleration[0] = result.x();
    acceleration[1] = result.y();
    acceleration[2] = result.z();
    return 0;
#else
    (void)vertices_xyz; (void)vertex_count; (void)facets; (void)facet_count;
    (void)mu_body; (void)position; (void)acceleration; (void)potential;
    return -1;
#endif
}

void ryugu_werner_reset_cache(void) {
#if defined(RYUGU_BASILISK_NATIVE)
    cached_polyhedron.reset();
#endif
}

int32_t ryugu_exafmm_eval(const double *source_xyz,
                          const double *source_mass,
                          uint64_t source_count,
                          const double *target_xyz,
                          uint64_t target_count,
                          int32_t expansion_order,
                          int32_t leaf_capacity,
                          double *potential,
                          double *gradient_xyz) {
#if defined(RYUGU_EXAFMM_NATIVE)
    if (source_xyz == nullptr || source_mass == nullptr || target_xyz == nullptr
        || potential == nullptr || gradient_xyz == nullptr || source_count == 0 || target_count == 0) {
        return -2;
    }
    using namespace exafmm_t;
    Bodies<real_t> sources(source_count), targets(target_count);
    for (uint64_t index = 0; index < source_count; ++index) {
        sources[index].ibody = static_cast<int>(index);
        sources[index].q = source_mass[index];
        for (int component = 0; component < 3; ++component) {
            sources[index].X[component] = source_xyz[3 * index + component];
        }
    }
    for (uint64_t index = 0; index < target_count; ++index) {
        targets[index].ibody = static_cast<int>(index);
        for (int component = 0; component < 3; ++component) {
            targets[index].X[component] = target_xyz[3 * index + component];
        }
    }
    LaplaceFmm fmm(expansion_order, leaf_capacity);
    NodePtrs<real_t> leafs, nonleafs;
    Nodes<real_t> nodes;
    get_bounds(sources, targets, fmm.x0, fmm.r0);
    nodes = build_tree(sources, targets, leafs, nonleafs, fmm);
    init_rel_coord();
    build_list(nodes, fmm);
    fmm.M2L_setup(nonleafs);
    fmm.precompute();
    fmm.upward_pass(nodes, leafs);
    fmm.downward_pass(nodes, leafs);
    for (Node<real_t> *leaf : leafs) {
        for (size_t local = 0; local < leaf->itrgs.size(); ++local) {
            const int original = leaf->itrgs[local];
            potential[original] = 4.0 * kPi * kGravityConstant * leaf->trg_value[4 * local + 0];
            for (int component = 0; component < 3; ++component) {
                gradient_xyz[3 * original + component] =
                    4.0 * kPi * kGravityConstant * leaf->trg_value[4 * local + component + 1];
            }
        }
    }
    return 0;
#else
    (void)source_xyz; (void)source_mass; (void)source_count; (void)target_xyz;
    (void)target_count; (void)expansion_order; (void)leaf_capacity;
    (void)potential; (void)gradient_xyz;
    return -1;
#endif
}

int32_t ryugu_flups_free_space_eval(const double *density,
                                    int32_t nx,
                                    int32_t ny,
                                    int32_t nz,
                                    const double spacing[3],
                                    const double box_length[3],
                                    double gravitational_constant,
                                    double *potential,
                                    double *acceleration_xyz) {
#if defined(RYUGU_FLUPS_NATIVE)
    if (density == nullptr || spacing == nullptr || box_length == nullptr
        || potential == nullptr || acceleration_xyz == nullptr || nx <= 1 || ny <= 1 || nz <= 1) {
        return -2;
    }
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    const int global_size[3] = {nx, ny, nz};
    const int process_grid[3] = {1, 1, 1};
    FLUPS_Topology *topology = flups_topo_new(0, 1, global_size, process_grid, false,
                                               nullptr, 64, MPI_COMM_SELF);
    if (topology == nullptr) return -3;
    FLUPS_BoundaryType *boundary[3][2] = {};
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            boundary[axis][side] = static_cast<FLUPS_BoundaryType *>(
                flups_malloc(sizeof(FLUPS_BoundaryType)));
            boundary[axis][side][0] = UNB;
        }
    }
    const FLUPS_CenterType centers[3] = {CELL_CENTER, CELL_CENTER, CELL_CENTER};
    FLUPS_Solver *solver = flups_init_timed(topology, boundary, spacing, box_length,
                                             NOD, centers, nullptr);
    if (solver == nullptr) return -4;
    flups_set_greenType(solver, CHAT_2);
    flups_setup(solver, false);
    const size_t padded_count = flups_topo_get_memsize(topology);
    auto *rhs = static_cast<double *>(flups_malloc(padded_count * sizeof(double)));
    auto *field = static_cast<double *>(flups_malloc(padded_count * sizeof(double)));
    if (!rhs || !field) {
        flups_free(rhs); flups_free(field);
        flups_cleanup(solver); flups_topo_free(topology);
        for (auto &axis : boundary) for (auto *side : axis) flups_free(side);
        return -4;
    }
    std::fill(rhs, rhs + padded_count, 0.0);
    std::fill(field, field + padded_count, 0.0);
    const int nmem[3] = {flups_topo_get_nmem(topology, 0), flups_topo_get_nmem(topology, 1), flups_topo_get_nmem(topology, 2)};
    const int axis = flups_topo_get_axis(topology);
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        const size_t packed = (static_cast<size_t>(z) * ny + y) * nx + x;
        rhs[flups_locID(0, x, y, z, 0, axis, nmem, 1)] = 4.0 * kPi * gravitational_constant * density[packed];
    }
    flups_solve(solver, field, rhs, STD);
    for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        potential[(static_cast<size_t>(z) * ny + y) * nx + x] = field[flups_locID(0, x, y, z, 0, axis, nmem, 1)];
    }
    flups_free(rhs); flups_free(field);
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const size_t index = static_cast<size_t>(z) * ny * nx + static_cast<size_t>(y) * nx + x;
                const auto at = [&](int ix, int iy, int iz) {
                    ix = std::clamp(ix, 0, nx - 1);
                    iy = std::clamp(iy, 0, ny - 1);
                    iz = std::clamp(iz, 0, nz - 1);
                    return potential[static_cast<size_t>(iz) * ny * nx + static_cast<size_t>(iy) * nx + ix];
                };
                acceleration_xyz[3 * index + 0] = -(at(x + 1, y, z) - at(x - 1, y, z))
                    / (2.0 * spacing[0]);
                acceleration_xyz[3 * index + 1] = -(at(x, y + 1, z) - at(x, y - 1, z))
                    / (2.0 * spacing[1]);
                acceleration_xyz[3 * index + 2] = -(at(x, y, z + 1) - at(x, y, z - 1))
                    / (2.0 * spacing[2]);
            }
        }
    }
    flups_cleanup(solver);
    flups_topo_free(topology);
    for (auto &axis : boundary) for (auto *side : axis) flups_free(side);
    return 0;
#else
    (void)density; (void)nx; (void)ny; (void)nz; (void)spacing; (void)box_length;
    (void)gravitational_constant; (void)potential; (void)acceleration_xyz;
    return -1;
#endif
}
