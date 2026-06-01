/**
 * Timestamp helpers — one place to format and parse the timestamps Ragger
 * writes to the database.
 *
 * Ragger is a single-user, single-machine app, so database timestamps are the
 * user's LOCAL time in the fixed-width MySQL-style format "%F %T"
 * ("YYYY-MM-DD HH:MM:SS"). Fixed width means lexical order == chronological
 * order, so SQL comparisons (e.g. retention cutoffs) work on the raw string.
 *
 * Every timestamp that lands in a DB column should come from db_timestamp(),
 * and anything reading one back should parse it with parse_db_timestamp() —
 * the write and read formats must stay in lockstep.
 */
#pragma once

#include <ctime>
#include <optional>
#include <string>

namespace ragger {

/// Current local time as "YYYY-MM-DD HH:MM:SS" (strftime "%F %T").
/// The canonical format for every timestamp stored in the database.
std::string db_timestamp();

/// Format an explicit instant as local "YYYY-MM-DD HH:MM:SS".
std::string db_timestamp(std::time_t when);

/// Parse a db_timestamp() string back to a local time_t.
/// Returns nullopt if the string is not in the expected format.
std::optional<std::time_t> parse_db_timestamp(const std::string& s);

/// Compact local stamp for filenames: "YYYYMMDD_HHMMSS_mmm" (millisecond
/// precision). Used for payload-dump filenames, never stored in the DB.
std::string file_stamp();

} // namespace ragger
