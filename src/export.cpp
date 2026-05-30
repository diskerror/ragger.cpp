/**
 * Export — mysqldump-style SQL dump implementation.
 *
 * Opens the database READONLY so it's safe to run against a live server.
 * Dumps CREATE TABLE / CREATE INDEX / CREATE TRIGGER + INSERT statements.
 * The `embedding` BLOB column in `memories` is skipped unless requested.
 */
#include "ragger/export.h"
#include "ragger/config.h"
#include "ragger/lang.h"

#include <sqlite3.h>

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

static std::string blob_hex(const void* data, int len) {
    static const char hex[] = "0123456789ABCDEF";
    std::string out = "X'";
    auto* p = static_cast<const unsigned char*>(data);
    for (int i = 0; i < len; ++i) {
        out += hex[p[i] >> 4];
        out += hex[p[i] & 0x0F];
    }
    out += '\'';
    return out;
}

// Columns to skip in a given table unless embeddings are requested.
// The v2 content tables (turns/summaries/decisions/documents) each carry an
// `embedding` BLOB; skip it by default to keep dumps small and diffable.
static bool skip_column(const std::string& /*table*/,
                        const std::string& column,
                        bool include_embeddings) {
    return !include_embeddings && column == "embedding";
}

// Tables that are purely internal / rebuilt on startup. FTS5 maintains its
// own shadow tables (<name>_data/_idx/_docsize/_config) under each *_fts
// virtual table; those are derived content and are rebuilt from the base
// tables, so they're excluded from dumps.
static bool is_internal_table(const std::string& name) {
    return name.find("_fts") != std::string::npos;
}

// -- public API -------------------------------------------------------------

std::vector<std::string> export_list_tables(const std::string& db_path) {
    std::string resolved = expand_path(db_path);
    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(resolved.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
        std::string err = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
    }

    std::vector<std::string> tables;
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db,
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name",
        -1, &stmt, nullptr);
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (!is_internal_table(name))
            tables.push_back(name);
    }
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return tables;
}

int export_sql(std::ostream& out,
               const std::string& db_path,
               const ExportOptions& opts) {

    std::string resolved = expand_path(db_path);
    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(resolved.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
        std::string err = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
    }

    // Collect schema objects (tables, indexes, triggers) from sqlite_master.
    struct SchemaObj { std::string type; std::string name; std::string tbl_name; std::string sql; };
    std::vector<SchemaObj> schema;
    {
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY CASE type "
            "  WHEN 'table'   THEN 0 "
            "  WHEN 'index'   THEN 1 "
            "  WHEN 'trigger' THEN 2 "
            "  ELSE 3 END, name",
            -1, &stmt, nullptr);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            SchemaObj obj;
            obj.type     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            obj.name     = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            obj.tbl_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            obj.sql      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            schema.push_back(std::move(obj));
        }
        sqlite3_finalize(stmt);
    }

    // Filter to requested table (if any).
    auto table_match = [&](const std::string& tbl) {
        if (opts.table.empty()) return !is_internal_table(tbl);
        return tbl == opts.table;
    };

    out << "-- Ragger database export\n";
    out << "PRAGMA foreign_keys = OFF;\n";
    out << "BEGIN TRANSACTION;\n\n";

    int total_rows = 0;

    for (auto& obj : schema) {
        if (!table_match(obj.tbl_name)) continue;

        // Schema DDL
        if (obj.type == "table") {
            out << obj.sql << ";\n";
        } else {
            out << obj.sql << ";\n";
        }
    }
    out << "\n";

    // Data: INSERT statements for each matching table.
    for (auto& obj : schema) {
        if (obj.type != "table") continue;
        if (!table_match(obj.name)) continue;

        // Get column info via PRAGMA.
        struct ColInfo { std::string name; };
        std::vector<ColInfo> all_cols;
        {
            std::string pragma = "PRAGMA table_info(" + obj.name + ")";
            sqlite3_stmt* info = nullptr;
            sqlite3_prepare_v2(db, pragma.c_str(), -1, &info, nullptr);
            while (sqlite3_step(info) == SQLITE_ROW) {
                ColInfo ci;
                ci.name = reinterpret_cast<const char*>(sqlite3_column_text(info, 1));
                all_cols.push_back(ci);
            }
            sqlite3_finalize(info);
        }

        // Decide which columns to emit.
        std::vector<int> col_indices;
        std::vector<std::string> col_names;
        for (int i = 0; i < (int)all_cols.size(); ++i) {
            if (skip_column(obj.name, all_cols[i].name, opts.include_embeddings))
                continue;
            col_indices.push_back(i);
            col_names.push_back(all_cols[i].name);
        }

        // SELECT all rows.
        std::string select = "SELECT * FROM " + obj.name;
        sqlite3_stmt* data = nullptr;
        sqlite3_prepare_v2(db, select.c_str(), -1, &data, nullptr);

        while (sqlite3_step(data) == SQLITE_ROW) {
            out << "INSERT INTO " << obj.name << " (";
            for (size_t j = 0; j < col_names.size(); ++j) {
                if (j > 0) out << ", ";
                out << col_names[j];
            }
            out << ") VALUES (";

            for (size_t j = 0; j < col_indices.size(); ++j) {
                if (j > 0) out << ", ";
                int ci = col_indices[j];
                int coltype = sqlite3_column_type(data, ci);
                switch (coltype) {
                    case SQLITE_NULL:
                        out << "NULL";
                        break;
                    case SQLITE_INTEGER:
                        out << sqlite3_column_int64(data, ci);
                        break;
                    case SQLITE_FLOAT:
                        out << sqlite3_column_double(data, ci);
                        break;
                    case SQLITE_BLOB:
                        out << blob_hex(sqlite3_column_blob(data, ci),
                                        sqlite3_column_bytes(data, ci));
                        break;
                    case SQLITE_TEXT:
                    default:
                        out << sql_quote(reinterpret_cast<const char*>(
                                sqlite3_column_text(data, ci)));
                        break;
                }
            }
            out << ");\n";
            ++total_rows;
        }
        sqlite3_finalize(data);
    }

    out << "\nCOMMIT;\n";

    sqlite3_close(db);
    return total_rows;
}

} // namespace ragger
