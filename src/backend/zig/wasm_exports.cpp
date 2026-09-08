#include <cstdlib>
#include <cstdint>

extern "C" {
__attribute__((visibility("default"))) void *ryugu_alloc(uint32_t bytes) {
    return std::malloc(bytes);
}
__attribute__((visibility("default"))) void ryugu_free(void *pointer) {
    std::free(pointer);
}
}
