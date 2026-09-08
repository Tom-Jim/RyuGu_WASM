#ifndef RYUGU_BASILISK_BRIDGE_H
#define RYUGU_BASILISK_BRIDGE_H

#include <stdint.h>

#if defined(__clang__)
#pragma clang visibility push(default)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RyuguBasiliskState {
    double position_m[3];
    double velocity_mps[3];
    uint64_t simulation_time_ns;
    uint64_t epoch;
    uint64_t sequence;
} RyuguBasiliskState;

typedef struct RyuguBasiliskForce {
    double acceleration_mps2[3];
    double potential_m2ps2;
    uint8_t algorithm;
    uint8_t reserved[7];
} RyuguBasiliskForce;

uint16_t ryugu_basilisk_protocol_version(void);

// Legacy placeholder retained for reference only.
// int32_t ryugu_basilisk_step(const RyuguBasiliskState *state,
//                           const RyuguBasiliskForce *force, uint64_t step_ns);
int32_t ryugu_scheduler_reset(uint64_t period_ns);
int32_t ryugu_scheduler_advance(uint64_t stop_ns);
uint64_t ryugu_scheduler_time(void);

int32_t ryugu_direct_sum_eval(const double *source_xyz,
                              const double *source_mass,
                              uint64_t source_count,
                              const double position_m[3],
                              double acceleration_mps2[3],
                              double *potential_m2ps2);

int32_t ryugu_radial_boost_eval(const double *cells,
                                uint64_t cell_count,
                                const double position_m[3],
                                double acceleration_mps2[3],
                                double *potential_m2ps2,
                                double abs_tolerance,
                                double rel_tolerance);

int32_t ryugu_werner_eval(const double *vertices_xyz,
                          uint64_t vertex_count,
                          const uint32_t *facets,
                          uint64_t facet_count,
                          double mu_body,
                          const double position_m[3],
                          double acceleration_mps2[3],
                          double *potential_m2ps2);
void ryugu_werner_reset_cache(void);

int32_t ryugu_exafmm_eval(const double *source_xyz,
                          const double *source_mass,
                          uint64_t source_count,
                          const double *target_xyz,
                          uint64_t target_count,
                          int32_t expansion_order,
                          int32_t leaf_capacity,
                          double *potential,
                          double *gradient_xyz);

int32_t ryugu_flups_free_space_eval(const double *density,
                                    int32_t nx,
                                    int32_t ny,
                                    int32_t nz,
                                    const double spacing_m[3],
                                    const double box_length_m[3],
                                    double gravitational_constant,
                                    double *potential,
                                    double *acceleration_xyz);

#ifdef __cplusplus
}
#endif

#if defined(__clang__)
#pragma clang visibility pop
#endif

#endif
