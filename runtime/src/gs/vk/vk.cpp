#include "vk.hpp"

#include <cstdlib>
#include <cstring>

namespace ps2::gs::vk {

bool wanted() {
    const char *backend = std::getenv("PS2_GS");
    return backend && std::strcmp(backend, "hw") == 0;
}

}
