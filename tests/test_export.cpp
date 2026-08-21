/**
 * Export tests — exercises SQL dump output against a temp SQLite database.
 * No ONNX model needed; uses SqliteBackend's DB-only constructor + raw SQL.
 */
#include "export.h"
#include "config.h"

#include <sqlite3.h>

#include <cassert>
#include <filesystem>
#include <print>
#include <sstream>

namespace fs = std::filesystem;

static const std::string TEMP_DB = "/tmp/ragger_test_export.db";

static void cleanup() {
    fs::remove(TEMP_DB);
    fs::remove(TEMP_DB + "-wal");
    fs::remove(TEMP_DB + "-shm");
}

static void exec(sqlite3* db, const char* sql) {
    char* err = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string msg = err ? err : "unknown";
        sqlite3_free(err);
        throw std::runtime_error(msg);
    }
}

static sqlite3* create_test_db() {
    cleanup();
    sqlite3* db = nullptr;
    sqlite3_open(TEMP_DB.c_str(), &db);
    exec(db, R"(
        CREATE TABLE memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            text      TEXT NOT NULL,
            embedding BLOB,
            metadata  TEXT,
            timestamp TEXT NOT NULL
        )
    )");
    exec(db, R"(
        CREATE TABLE settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    )");
    exec(db, "INSERT INTO memories (text, embedding, metadata, timestamp) "
             "VALUES ('hello world', X'DEADBEEF', '{\"k\":\"v\"}', '2026-01-01T00:00:00Z')");
    exec(db, "INSERT INTO memories (text, embedding, metadata, timestamp) "
             "VALUES ('it''s a test', NULL, '{}', '2026-06-15T12:00:00Z')");
    exec(db, "INSERT INTO settings (key, value) VALUES ('version', '1')");
    return db;
}

// -----------------------------------------------------------------------

void test_export_all_no_embeddings() {
    std::println("  test_export_all_no_embeddings...");
    auto* db = create_test_db();
    sqlite3_close(db);

    std::ostringstream out;
    ragger::ExportOptions opts;
    int rows = ragger::export_sql(out, TEMP_DB, opts);
    std::string sql = out.str();

    assert(rows == 3);
    assert(sql.find("BEGIN TRANSACTION") != std::string::npos);
    assert(sql.find("COMMIT") != std::string::npos);
    assert(sql.find("CREATE TABLE memories") != std::string::npos);
    assert(sql.find("CREATE TABLE settings") != std::string::npos);
    // embedding column should NOT appear in INSERT column lists
    assert(sql.find("INSERT INTO memories (id, text, metadata, timestamp)") != std::string::npos);
    assert(sql.find("DEADBEEF") == std::string::npos);
    // Data values present
    assert(sql.find("hello world") != std::string::npos);
    assert(sql.find("it''s a test") != std::string::npos);

    cleanup();
    std::println("    OK");
}

void test_export_all_with_embeddings() {
    std::println("  test_export_all_with_embeddings...");
    auto* db = create_test_db();
    sqlite3_close(db);

    std::ostringstream out;
    ragger::ExportOptions opts;
    opts.include_embeddings = true;
    int rows = ragger::export_sql(out, TEMP_DB, opts);
    std::string sql = out.str();

    assert(rows == 3);
    assert(sql.find("INSERT INTO memories (id, text, embedding, metadata, timestamp)") != std::string::npos);
    assert(sql.find("DEADBEEF") != std::string::npos);

    cleanup();
    std::println("    OK");
}

void test_export_single_table() {
    std::println("  test_export_single_table...");
    auto* db = create_test_db();
    sqlite3_close(db);

    std::ostringstream out;
    ragger::ExportOptions opts;
    opts.table = "settings";
    int rows = ragger::export_sql(out, TEMP_DB, opts);
    std::string sql = out.str();

    assert(rows == 1);
    assert(sql.find("CREATE TABLE settings") != std::string::npos);
    assert(sql.find("CREATE TABLE memories") == std::string::npos);
    assert(sql.find("INSERT INTO settings") != std::string::npos);

    cleanup();
    std::println("    OK");
}

void test_export_list_tables() {
    std::println("  test_export_list_tables...");
    auto* db = create_test_db();
    sqlite3_close(db);

    auto tables = ragger::export_list_tables(TEMP_DB);
    assert(tables.size() == 2);
    bool has_memories = false, has_settings = false;
    for (auto& t : tables) {
        if (t == "memories") has_memories = true;
        if (t == "settings") has_settings = true;
    }
    assert(has_memories);
    assert(has_settings);

    cleanup();
    std::println("    OK");
}

void test_export_null_handling() {
    std::println("  test_export_null_handling...");
    auto* db = create_test_db();
    sqlite3_close(db);

    std::ostringstream out;
    ragger::ExportOptions opts;
    opts.include_embeddings = true;
    ragger::export_sql(out, TEMP_DB, opts);
    std::string sql = out.str();

    // Second row has NULL embedding
    assert(sql.find("NULL") != std::string::npos);

    cleanup();
    std::println("    OK");
}

void test_export_roundtrip() {
    std::println("  test_export_roundtrip...");
    auto* db = create_test_db();
    sqlite3_close(db);

    // Export
    std::ostringstream out;
    ragger::ExportOptions opts;
    opts.include_embeddings = true;
    ragger::export_sql(out, TEMP_DB, opts);
    std::string sql = out.str();

    // Import into a fresh DB
    std::string rt_db = "/tmp/ragger_test_export_rt.db";
    fs::remove(rt_db);
    sqlite3* db2 = nullptr;
    sqlite3_open(rt_db.c_str(), &db2);
    exec(db2, sql.c_str());

    // Verify row count
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db2, "SELECT COUNT(*) FROM memories", -1, &stmt, nullptr);
    sqlite3_step(stmt);
    int count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    assert(count == 2);

    sqlite3_prepare_v2(db2, "SELECT COUNT(*) FROM settings", -1, &stmt, nullptr);
    sqlite3_step(stmt);
    count = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
    assert(count == 1);

    sqlite3_close(db2);
    fs::remove(rt_db);
    cleanup();
    std::println("    OK");
}

// -----------------------------------------------------------------------

int main() {
    ragger::init_config("");

    std::println("Running export tests:");
    test_export_all_no_embeddings();
    test_export_all_with_embeddings();
    test_export_single_table();
    test_export_list_tables();
    test_export_null_handling();
    test_export_roundtrip();

    std::println("test_export: all passed");
    return 0;
}
