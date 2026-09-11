/**
 * sqlite_text_index.cpp — SQLite storage hooks for the custom terms index.
 */
#include "sqlite_text_index.h"

#include <format>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "util/sqlite.h"

namespace ragger {

// Doc embed text formula (title appended when present) — mirrors the constant
// SqliteBackend uses for embeddings so the index sees the same document text.
namespace {
constexpr const char* kDocEmbedTextSQL =
    "text || CASE WHEN (SELECT ifnull(title,'') FROM document_sources ds "
    "WHERE ds.document_source_id = documents.document_source_id) = '' "
    "THEN '' ELSE char(10) || (SELECT title FROM document_sources ds "
    "WHERE ds.document_source_id = documents.document_source_id) END";
}  // namespace

void SqliteTextIndex::validate_table(const std::string& table) {
    static const std::unordered_set<std::string> valid = {
        "turns", "turn_summaries", "summaries", "documents", "decisions"};
    if (valid.find(table) == valid.end())
        throw std::runtime_error("Invalid table name: " + table);
}

std::string SqliteTextIndex::id_col_of(const std::string& table) {
    if (table == "turns")          return "turn_id";
    if (table == "turn_summaries") return "turn_summary_id";
    if (table == "summaries")      return "summary_id";
    if (table == "documents")      return "document_id";
    if (table == "decisions")      return "decision_id";
    throw std::runtime_error("Invalid table name: " + table);
}

std::string SqliteTextIndex::text_select_of(const std::string& table) {
    const std::string id = id_col_of(table);
    if (table == "documents")
        return std::format("SELECT {}, {} AS text FROM documents ORDER BY {}",
                           id, kDocEmbedTextSQL, id);
    if (table == "turns")
        return "SELECT turn_id, user_text, assistant_text FROM turns ORDER BY turn_id";
    return std::format("SELECT {}, text FROM {} ORDER BY {}", id, table, id);
}

void SqliteTextIndex::create_schema() {
    auto exec = [&](const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "unknown error";
            sqlite3_free(err);
            throw std::runtime_error("SqliteTextIndex schema: " + msg);
        }
    };

    exec(R"(
        CREATE TABLE IF NOT EXISTS terms (
            term_id INTEGER PRIMARY KEY AUTOINCREMENT,
            term    TEXT NOT NULL UNIQUE
        )
    )");

    // One junction table per text table. FK ON DELETE CASCADE means a deleted
    // record's index rows go away automatically — nothing to maintain in the
    // write path beyond delete-then-insert for updates.
    struct T { const char* junc; const char* base; const char* pk; };
    const T tables[] = {
        {"turns_terms",          "turns",          "turn_id"},
        {"turn_summaries_terms", "turn_summaries", "turn_summary_id"},
        {"summaries_terms",      "summaries",      "summary_id"},
        {"documents_terms",      "documents",      "document_id"},
        {"decisions_terms",      "decisions",      "decision_id"},
    };
    for (const auto& t : tables) {
        exec(std::format(
            "CREATE TABLE IF NOT EXISTS {} (\n"
            "    {} INTEGER NOT NULL REFERENCES {}({}) ON DELETE CASCADE,\n"
            "    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,\n"
            "    count   INTEGER NOT NULL DEFAULT 1,\n"
            "    PRIMARY KEY ({}, term_id)\n)",
            t.junc, t.pk, t.base, t.pk, t.pk));
        exec(std::format(
            "CREATE INDEX IF NOT EXISTS idx_{}_term ON {}(term_id)",
            t.junc, t.junc));
    }
}

int SqliteTextIndex::upsert_term(const std::string& term) {
    {
        Stmt ins(db_, "INSERT OR IGNORE INTO terms(term) VALUES (?)");
        ins.bind(1, term);
        ins.exec();
    }
    Stmt sel(db_, "SELECT term_id FROM terms WHERE term = ?");
    sel.bind(1, term);
    if (sel.step()) return sel.column_int(0);
    return 0;
}

void SqliteTextIndex::replace_record_terms(
    const std::string& table, int id,
    const std::unordered_map<int, int>& term_counts) {
    validate_table(table);
    const std::string junc = table + "_terms";
    const std::string id_col = id_col_of(table);
    {
        Stmt del(db_, std::format("DELETE FROM {} WHERE {} = ?", junc, id_col));
        del.bind(1, id);
        del.exec();
    }
    for (const auto& [term_id, count] : term_counts) {
        Stmt ins(db_, std::format(
            "INSERT OR REPLACE INTO {} ({}, term_id, count) VALUES (?, ?, ?)",
            junc, id_col));
        ins.bind(1, id).bind(2, term_id).bind(3, count);
        ins.exec();
    }
}

void SqliteTextIndex::set_counts(const std::string& table, int id,
                                 int unigram_count, int bigram_count) {
    validate_table(table);
    Stmt upd(db_, std::format(
        "UPDATE {} SET unigram_count = ?, bigram_count = ? WHERE {} = ?",
        table, id_col_of(table)));
    upd.bind(1, unigram_count).bind(2, bigram_count).bind(3, id);
    upd.exec();
}

int SqliteTextIndex::corpus_size(const std::string& table) {
    validate_table(table);
    Stmt s(db_, std::format("SELECT COUNT(*) FROM {}", table));
    return s.step() ? s.column_int(0) : 0;
}

int SqliteTextIndex::doc_freq(const std::string& table, int term_id) {
    validate_table(table);
    Stmt s(db_, std::format("SELECT COUNT(*) FROM {}_terms WHERE term_id = ?", table));
    s.bind(1, term_id);
    return s.step() ? s.column_int(0) : 0;
}

int SqliteTextIndex::lookup_term(const std::string& term) {
    Stmt s(db_, "SELECT term_id FROM terms WHERE term = ?");
    s.bind(1, term);
    return s.step() ? s.column_int(0) : 0;
}

std::vector<TextIndex::Posting> SqliteTextIndex::postings(
    const std::string& table, int term_id) {
    validate_table(table);
    const std::string id_col = id_col_of(table);
    Stmt s(db_,
        "SELECT j." + id_col + ", j.count, b.unigram_count, b.bigram_count "
        "FROM " + table + "_terms j JOIN " + table + " b "
        "ON b." + id_col + " = j." + id_col + " "
        "WHERE j.term_id = ?");
    s.bind(1, term_id);
    std::vector<Posting> out;
    while (s.step()) {
        out.push_back({s.column_int(0), s.column_int(1),
                       s.column_int(2), s.column_int(3)});
    }
    return out;
}

int SqliteTextIndex::reindex_table(const std::string& table, bool progress) {
    validate_table(table);
    const std::string id_col = id_col_of(table);
    const std::string select_sql = text_select_of(table);

    int total = 0;
    if (progress) {
        Stmt c(db_, std::format("SELECT COUNT(*) FROM {}", table));
        if (c.step()) total = c.column_int(0);
    }

    // Two-pass batches: read a bounded batch (finalize the SELECT), then write.
    // Walking a SELECT while UPDATE-ing the same table on one connection can
    // invalidate the cursor mid-scan (see rebuild_phon's note).
    constexpr int kBatch = 200;
    struct Row { int id; std::string text; };
    int done = 0;
    int last_id = 0;

    for (;;) {
        std::vector<Row> batch;
        batch.reserve(kBatch);
        {
            // select_sql is ORDER BY id; skip ids already processed to page.
            Stmt s(db_, select_sql);
            while (s.step()) {
                int id = s.column_int(0);
                if (id <= last_id) continue;
                std::string text = s.column_text(1);
                if (table == "turns") {
                    std::string a = s.column_text(2);
                    if (!a.empty()) text += "\n" + a;
                }
                batch.push_back({id, std::move(text)});
                if (static_cast<int>(batch.size()) >= kBatch) break;
            }
        }
        if (batch.empty()) break;
        for (const auto& row : batch) {
            index_record(table, row.id, row.text);
            last_id = row.id;
            ++done;
            if (progress) {
                std::cout << std::format("\rReindexing {}: {}/{}",
                                         table, done, total);
                std::cout.flush();
            }
        }
    }
    if (progress) std::cout << "\n";
    return done;
}

}  // namespace ragger
