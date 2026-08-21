/**
 * Export — mysqldump-style SQL dump of the Ragger database.
 *
 * Opens the DB read-only; safe to run while the server is up.
 */
#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace ragger {

struct ExportOptions {
    bool include_embeddings = false;
    std::string table;               // empty = all tables
};

/// Dump the database at `db_path` as SQL to `out`.
/// Returns the number of rows written.
int export_sql(std::ostream& out,
               const std::string& db_path,
               const ExportOptions& opts);

/// List user-visible table names in the database.
std::vector<std::string> export_list_tables(const std::string& db_path);

} // namespace ragger
