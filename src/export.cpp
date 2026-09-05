/**
 * Export — mysqldump-style SQL dump implementation.
 *
 * The StorageBackend-based overloads are the canonical implementation.
 * The db_path convenience overloads open a read-only SqliteBackend and
 * delegate — no CREATE TABLE side-effects on arbitrary dump targets.
 * Dumps CREATE TABLE / CREATE INDEX / CREATE TRIGGER + INSERT statements.
 * The `embedding` BLOB column in content tables is skipped unless requested.
 */
#include "export.h"
#include "storage_backend.h"
#include "sqlite_backend.h"
#include "config.h"

#include <format>
#include <stdexcept>
#include <vector>

namespace ragger {

// -- helpers ----------------------------------------------------------------

static std::string sql_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "''";
        else           out += c;
    }
    out += '\'';
    return out;
}

static std::string blob_hex(const std::vector<uint8_t>& data) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out = "X'";
    for (uint8_t b : data) {
        out += hex[b >> 4];
        out += hex[b & 0x0F];
    }
    out += '\'';
    return out;
}

// Columns to skip in a given table unless embeddings are requested.
static bool skip_column(const std::string& /*table*/,
                        const std::string& column,
                        bool include_embeddings) {
    return !include_embeddings && column == "embedding";
}

// Tables that are purely internal / rebuilt on startup. FTS5 shadow tables
// are derived content excluded from dumps. (list_schema_objects already
// filters _fts shadow tables; is_internal_table guards tbl_name matching.)
static bool is_internal_table(const std::string& name) {
    return name.find("_fts") != std::string::npos;
}

// -- StorageBackend-based implementations -----------------------------------

std::vector<std::string> export_list_tables(StorageBackend& backend) {
    std::vector<std::string> tables;
    for (auto& obj : backend.list_schema_objects()) {
        if (obj.type == "table" && !is_internal_table(obj.name))
            tables.push_back(obj.name);
    }
    return tables;
}

int export_sql(std::ostream& out, StorageBackend& backend, const ExportOptions& opts) {
    auto schema = backend.list_schema_objects();

    auto table_match = [&](const std::string& tbl) {
        if (opts.table.empty()) return !is_internal_table(tbl);
        return tbl == opts.table;
    };

    out << "-- Ragger database export\n";
    out << "PRAGMA foreign_keys = OFF;\n";
    out << "BEGIN TRANSACTION;\n\n";

    int total_rows = 0;

    // Schema DDL
    for (auto& obj : schema) {
        if (!table_match(obj.tbl_name)) continue;
        out << obj.sql << ";\n";
    }
    out << "\n";

    // Data: INSERT statements for each matching table.
    for (auto& obj : schema) {
        if (obj.type != "table") continue;
        if (!table_match(obj.name)) continue;

        auto all_cols = backend.table_column_names(obj.name);

        // Decide which columns to emit.
        std::vector<int>         col_indices;
        std::vector<std::string> col_names;
        for (int i = 0; i < (int)all_cols.size(); ++i) {
            if (skip_column(obj.name, all_cols[i], opts.include_embeddings))
                continue;
            col_indices.push_back(i);
            col_names.push_back(all_cols[i]);
        }

        backend.iterate_table_rows(obj.name, [&](const ExportRow& row) {
            out << "INSERT INTO " << obj.name << " (";
            for (size_t j = 0; j < col_names.size(); ++j) {
                if (j > 0) out << ", ";
                out << col_names[j];
            }
            out << ") VALUES (";

            for (size_t j = 0; j < col_indices.size(); ++j) {
                if (j > 0) out << ", ";
                int ci = col_indices[j];
                const auto& cell = row[ci];
                switch (cell.type) {
                    case ExportCell::Type::Null:
                        out << "NULL";
                        break;
                    case ExportCell::Type::Integer:
                        out << cell.int_val;
                        break;
                    case ExportCell::Type::Float:
                        out << cell.float_val;
                        break;
                    case ExportCell::Type::Blob:
                        out << blob_hex(cell.blob_val);
                        break;
                    case ExportCell::Type::Text:
                    default:
                        out << sql_quote(cell.text_val);
                        break;
                }
            }
            out << ");\n";
            ++total_rows;
        });
    }

    out << "\nCOMMIT;\n";
    return total_rows;
}

// -- db_path convenience overloads (thin wrappers) --------------------------

std::vector<std::string> export_list_tables(const std::string& db_path) {
    SqliteBackend backend(expand_path(db_path), /*readonly=*/true);
    return export_list_tables(backend);
}

int export_sql(std::ostream& out,
               const std::string& db_path,
               const ExportOptions& opts) {
    SqliteBackend backend(expand_path(db_path), /*readonly=*/true);
    return export_sql(out, backend, opts);
}

} // namespace ragger
