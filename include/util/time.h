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

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>

namespace ragger {

/// Current local time as "YYYY-MM-DD HH:MM:SS" (strftime "%F %T").
/// The canonical format for every timestamp stored in the database.
std::string db_timestamp();

/// Current time as Unix epoch seconds -- for INTEGER timestamp columns
/// (turns.created_at, summaries.created_at/updated_at, decisions.created_at,
/// models.created_at, sessions.created_at, users.created_at/updated_at).
/// See db_timestamp() for the TEXT-formatted equivalent used in
/// logging/display contexts -- the two are NOT interchangeable.
int64_t db_epoch();

/// Convert an already-resolved instant (e.g. from parse_db_timestamp) to
/// epoch seconds, for callers binding a specific historical timestamp
/// rather than "now".
int64_t db_epoch(std::time_t when);

/// Format an explicit instant as local "YYYY-MM-DD HH:MM:SS".
std::string db_timestamp(std::time_t when);

/// Parse a db_timestamp() string back to a local time_t.
/// Returns nullopt if the string is not in the expected format.
std::optional<std::time_t> parse_db_timestamp(const std::string& s);

/// Compact local stamp for filenames: "YYYYMMDD_HHMMSS_mmm" (millisecond
/// precision). Used for payload-dump filenames, never stored in the DB.
std::string file_stamp();

/// Local time as RFC 3339 with a SPACE separator and the UTC offset:
/// "YYYY-MM-DD HH:MM:SS±HH:MM" (e.g. "2026-11-13 00:30:12-08:00"). Chosen
/// because SQLite's unixepoch()/datetime() parse it straight back, so nothing
/// reading a stored value needs a custom parser (X16 clause 9: the
/// pre-migration backup timestamp). The two-argument form formats a fixed
/// instant with a fixed offset (tests pin it); the no-argument form uses the
/// current local time and its offset.
std::string local_time_string();
std::string local_time_string(std::time_t when, int offset_seconds);

} // namespace ragger
