#include "architecture/system_model/sim_model.h"
#include "architecture/messaging/messaging.h"
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include <memory>
#include <cstdint>

extern "C" __attribute__((import_module("env"), import_name("ryugu_rust_tick")))
void ryugu_rust_tick(uint64_t time_ns, double *state);

namespace {
class RustDynamics final : public SysModel {
public:
    Message<SCStatesMsgPayload> stateMessage;
    void UpdateState(uint64_t time_ns) override {
        double state[6] = {};
        ryugu_rust_tick(time_ns, state);
        SCStatesMsgPayload payload = {};
        for (int i = 0; i < 3; ++i) {
            payload.r_BN_N[i] = payload.r_CN_N[i] = state[i];
            payload.v_BN_N[i] = payload.v_CN_N[i] = state[i + 3];
        }
        stateMessage.write(&payload, moduleID, time_ns);
    }
};
struct Runtime {
    RustDynamics dynamics;
    SysModelTask task;
    SysProcess process;
    SimModel simulation;
    explicit Runtime(uint64_t period) : task(period), process("ryugu") {
        dynamics.ModelTag = "RustDynamics";
        task.TaskName = "physics";
        task.AddNewObject(&dynamics, 100);
        process.addNewTask(&task, 100);
        process.setPriority(100);
        process.enableProcess();
        simulation.addNewProcess(&process);
        simulation.selfInitSimulation();
        simulation.resetInitSimulation();
    }
};
std::unique_ptr<Runtime> runtime;
}

extern "C" {
int32_t ryugu_scheduler_reset(uint64_t period) {
    if (period == 0) return -2;
    runtime = std::make_unique<Runtime>(period);
    return 0;
}
int32_t ryugu_scheduler_advance(uint64_t stop) {
    if (!runtime || stop < runtime->simulation.CurrentNanos || stop % runtime->task.TaskPeriod) return -2;
    runtime->simulation.StepUntilStop(stop, -1);
    return runtime->simulation.CurrentNanos == stop ? 0 : -3;
}
uint64_t ryugu_scheduler_time() {
    return runtime ? runtime->simulation.CurrentNanos : 0;
}
}
