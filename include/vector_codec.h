// ragger/vector_codec.h — forwarding header to Diskerror::EmbeddingCodec.
// The canonical implementation lives in vendor/c_lib/EmbeddingCodec.h.
// This header provides the ragger::vector_codec namespace alias so existing
// code continues to compile without changes.
#pragma once

#include "EmbeddingCodec.h"

namespace ragger {
    namespace vector_codec = Diskerror::EmbeddingCodec;
}
