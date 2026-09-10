// ragger/stemmer.h — forwarding header to Diskerror::stem_en.
// The canonical implementation lives in vendor/c_lib/Stemmer.h.
// This header provides the ragger namespace wrapper so existing code
// continues to compile without changes.
#pragma once

#include "Stemmer.h"

namespace ragger {
    using Diskerror::stem_en;
}
