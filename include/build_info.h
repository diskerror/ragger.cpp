/**
 * Build metadata, generated at build time (cmake/build_info.cmake) into
 * build_info.cpp — only that file recompiles when the commit or version
 * changes, so the reported version is never stale.
 */
#pragma once

namespace ragger::build {
const char* version();
const char* commit();
const char* date();
}

#define RAGGER_VERSION    (::ragger::build::version())
#define RAGGER_COMMIT     (::ragger::build::commit())
#define RAGGER_BUILD_DATE (::ragger::build::date())
