// ragger/double_metaphone.h — forwarding header to Diskerror::DoubleMetaphone.
// The canonical implementation lives in vendor/c_lib/DoubleMetaphone.h.
// This header provides the ragger namespace wrappers so existing code
// continues to compile without changes.
#pragma once

#include "DoubleMetaphone.h"

namespace ragger {
    using Diskerror::double_metaphone;
    using Diskerror::phonize;
}
