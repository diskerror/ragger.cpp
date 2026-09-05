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

class StorageBackend;

struct ExportOptions {
    bool include_embeddings = false;
    std::string table;               // empty = all tables
};

/// List user-visible table names via a StorageBackend.
std::vector<std::string> export_list_tables(StorageBackend& backend);

/// Dump via a StorageBackend. Returns the number of rows written.
int export_sql(std::ostream& out, StorageBackend& backend, const ExportOptions& opts);

/// Convenience overloads: open the DB at db_path read-only, then delegate.
std::vector<std::string> export_list_tables(const std::string& db_path);
int export_sql(std::ostream& out,
               const std::string& db_path,
               const ExportOptions& opts);

} // namespace ragger
