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

// ---- cached-statement plumbing --------------------------------------------

Stmt& SqliteTextIndex::cached(std::optional<Stmt>& slot, const char* sql) {
    if (!slot) {
        slot.emplace(db_, sql);
    } else {
        sqlite3_reset(slot->raw());
        sqlite3_clear_bindings(slot->raw());
    }
    return *slot;
}

Stmt& SqliteTextIndex::select_term_id() {
    return cached(select_term_id_, "SELECT term_id FROM terms WHERE term = ?");
}
Stmt& SqliteTextIndex::insert_term() {
    return cached(insert_term_, "INSERT OR IGNORE INTO terms(term) VALUES (?)");
}

Stmt& SqliteTextIndex::doc_freq_turns() {
    return cached(doc_freq_turns_, "SELECT COUNT(*) FROM turns_terms WHERE term_id = ?");
}
Stmt& SqliteTextIndex::doc_freq_turn_summaries() {
    return cached(doc_freq_turn_summaries_,
                  "SELECT COUNT(*) FROM turn_summaries_terms WHERE term_id = ?");
}
Stmt& SqliteTextIndex::doc_freq_summaries() {
    return cached(doc_freq_summaries_,
                  "SELECT COUNT(*) FROM summaries_terms WHERE term_id = ?");
}
Stmt& SqliteTextIndex::doc_freq_documents() {
    return cached(doc_freq_documents_,
                  "SELECT COUNT(*) FROM documents_terms WHERE term_id = ?");
}
Stmt& SqliteTextIndex::doc_freq_decisions() {
    return cached(doc_freq_decisions_,
                  "SELECT COUNT(*) FROM decisions_terms WHERE term_id = ?");
}

Stmt& SqliteTextIndex::count_turns() {
    return cached(count_turns_, "SELECT COUNT(*) FROM turns");
}
Stmt& SqliteTextIndex::count_turn_summaries() {
    return cached(count_turn_summaries_, "SELECT COUNT(*) FROM turn_summaries");
}
Stmt& SqliteTextIndex::count_summaries() {
    return cached(count_summaries_, "SELECT COUNT(*) FROM summaries");
}
Stmt& SqliteTextIndex::count_documents() {
    return cached(count_documents_, "SELECT COUNT(*) FROM documents");
}
Stmt& SqliteTextIndex::count_decisions() {
    return cached(count_decisions_, "SELECT COUNT(*) FROM decisions");
}

Stmt& SqliteTextIndex::postings_turns() {
    return cached(postings_turns_,
        "SELECT j.turn_id, j.count, b.unigram_count, b.bigram_count "
        "FROM turns_terms j JOIN turns b ON b.turn_id = j.turn_id "
        "WHERE j.term_id = ?");
}
Stmt& SqliteTextIndex::postings_turn_summaries() {
    return cached(postings_turn_summaries_,
        "SELECT j.turn_summary_id, j.count, b.unigram_count, b.bigram_count "
        "FROM turn_summaries_terms j JOIN turn_summaries b "
        "ON b.turn_summary_id = j.turn_summary_id "
        "WHERE j.term_id = ?");
}
Stmt& SqliteTextIndex::postings_summaries() {
    return cached(postings_summaries_,
        "SELECT j.summary_id, j.count, b.unigram_count, b.bigram_count "
        "FROM summaries_terms j JOIN summaries b ON b.summary_id = j.summary_id "
        "WHERE j.term_id = ?");
}
Stmt& SqliteTextIndex::postings_documents() {
    return cached(postings_documents_,
        "SELECT j.document_id, j.count, b.unigram_count, b.bigram_count "
        "FROM documents_terms j JOIN documents b ON b.document_id = j.document_id "
        "WHERE j.term_id = ?");
}
Stmt& SqliteTextIndex::postings_decisions() {
    return cached(postings_decisions_,
        "SELECT j.decision_id, j.count, b.unigram_count, b.bigram_count "
        "FROM decisions_terms j JOIN decisions b ON b.decision_id = j.decision_id "
        "WHERE j.term_id = ?");
}

Stmt& SqliteTextIndex::delete_terms_turns() {
    return cached(delete_terms_turns_, "DELETE FROM turns_terms WHERE turn_id = ?");
}
Stmt& SqliteTextIndex::delete_terms_turn_summaries() {
    return cached(delete_terms_turn_summaries_,
                  "DELETE FROM turn_summaries_terms WHERE turn_summary_id = ?");
}
Stmt& SqliteTextIndex::delete_terms_summaries() {
    return cached(delete_terms_summaries_, "DELETE FROM summaries_terms WHERE summary_id = ?");
}
Stmt& SqliteTextIndex::delete_terms_documents() {
    return cached(delete_terms_documents_, "DELETE FROM documents_terms WHERE document_id = ?");
}
Stmt& SqliteTextIndex::delete_terms_decisions() {
    return cached(delete_terms_decisions_, "DELETE FROM decisions_terms WHERE decision_id = ?");
}

Stmt& SqliteTextIndex::insert_terms_turns() {
    return cached(insert_terms_turns_,
        "INSERT OR REPLACE INTO turns_terms (turn_id, term_id, count) VALUES (?, ?, ?)");
}
Stmt& SqliteTextIndex::insert_terms_turn_summaries() {
    return cached(insert_terms_turn_summaries_,
        "INSERT OR REPLACE INTO turn_summaries_terms (turn_summary_id, term_id, count) "
        "VALUES (?, ?, ?)");
}
Stmt& SqliteTextIndex::insert_terms_summaries() {
    return cached(insert_terms_summaries_,
        "INSERT OR REPLACE INTO summaries_terms (summary_id, term_id, count) VALUES (?, ?, ?)");
}
Stmt& SqliteTextIndex::insert_terms_documents() {
    return cached(insert_terms_documents_,
        "INSERT OR REPLACE INTO documents_terms (document_id, term_id, count) "
        "VALUES (?, ?, ?)");
}
Stmt& SqliteTextIndex::insert_terms_decisions() {
    return cached(insert_terms_decisions_,
        "INSERT OR REPLACE INTO decisions_terms (decision_id, term_id, count) "
        "VALUES (?, ?, ?)");
}

Stmt& SqliteTextIndex::update_counts_turns() {
    return cached(update_counts_turns_,
        "UPDATE turns SET unigram_count = ?, bigram_count = ? WHERE turn_id = ?");
}
Stmt& SqliteTextIndex::update_counts_turn_summaries() {
    return cached(update_counts_turn_summaries_,
        "UPDATE turn_summaries SET unigram_count = ?, bigram_count = ? "
        "WHERE turn_summary_id = ?");
}
Stmt& SqliteTextIndex::update_counts_summaries() {
    return cached(update_counts_summaries_,
        "UPDATE summaries SET unigram_count = ?, bigram_count = ? WHERE summary_id = ?");
}
Stmt& SqliteTextIndex::update_counts_documents() {
    return cached(update_counts_documents_,
        "UPDATE documents SET unigram_count = ?, bigram_count = ? WHERE document_id = ?");
}
Stmt& SqliteTextIndex::update_counts_decisions() {
    return cached(update_counts_decisions_,
        "UPDATE decisions SET unigram_count = ?, bigram_count = ? WHERE decision_id = ?");
}

// ---- runtime dispatch over the fixed 5-table set --------------------------

Stmt& SqliteTextIndex::doc_freq_stmt(const std::string& table) {
    if (table == "turns")          return doc_freq_turns();
    if (table == "turn_summaries") return doc_freq_turn_summaries();
    if (table == "summaries")      return doc_freq_summaries();
    if (table == "documents")      return doc_freq_documents();
    return doc_freq_decisions();
}
Stmt& SqliteTextIndex::count_stmt(const std::string& table) {
    if (table == "turns")          return count_turns();
    if (table == "turn_summaries") return count_turn_summaries();
    if (table == "summaries")      return count_summaries();
    if (table == "documents")      return count_documents();
    return count_decisions();
}
Stmt& SqliteTextIndex::postings_stmt(const std::string& table) {
    if (table == "turns")          return postings_turns();
    if (table == "turn_summaries") return postings_turn_summaries();
    if (table == "summaries")      return postings_summaries();
    if (table == "documents")      return postings_documents();
    return postings_decisions();
}
Stmt& SqliteTextIndex::delete_terms_stmt(const std::string& table) {
    if (table == "turns")          return delete_terms_turns();
    if (table == "turn_summaries") return delete_terms_turn_summaries();
    if (table == "summaries")      return delete_terms_summaries();
    if (table == "documents")      return delete_terms_documents();
    return delete_terms_decisions();
}
Stmt& SqliteTextIndex::insert_terms_stmt(const std::string& table) {
    if (table == "turns")          return insert_terms_turns();
    if (table == "turn_summaries") return insert_terms_turn_summaries();
    if (table == "summaries")      return insert_terms_summaries();
    if (table == "documents")      return insert_terms_documents();
    return insert_terms_decisions();
}
Stmt& SqliteTextIndex::update_counts_stmt(const std::string& table) {
    if (table == "turns")          return update_counts_turns();
    if (table == "turn_summaries") return update_counts_turn_summaries();
    if (table == "summaries")      return update_counts_summaries();
    if (table == "documents")      return update_counts_documents();
    return update_counts_decisions();
}

// ---- TextIndex overrides ---------------------------------------------------

void SqliteTextIndex::reset_terms_table() {
    // sqlite_sequence only has a row for `terms` once an AUTOINCREMENT insert
    // has happened; DELETE is a no-op (not an error) if the row is absent.
    char* err = nullptr;
    auto exec = [&](const char* sql) {
        if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "unknown error";
            sqlite3_free(err);
            throw std::runtime_error("SqliteTextIndex reset_terms_table: " + msg);
        }
    };
    // Cascades through every <table>_terms junction row via ON DELETE CASCADE
    // (foreign_keys must be ON -- SqliteBackend enables it at connection open).
    exec("DELETE FROM terms");
    exec("DELETE FROM sqlite_sequence WHERE name = 'terms'");
}

int SqliteTextIndex::upsert_term(const std::string& term) {
    // SELECT-first, not INSERT-OR-IGNORE-first: AUTOINCREMENT allocates the
    // next term_id BEFORE the uniqueness check runs, so an "ignored" insert
    // (term already exists) still permanently burns an id -- with upsert_term
    // called on every tokenize of every store/reindex/backfill, that leaked
    // ~13M ids against ~900K real terms in the live DB. Checking first avoids
    // ever attempting the redundant insert for an existing term.
    Stmt& sel = select_term_id();
    sel.bind(1, term);
    int existing = sel.step() ? sel.column_int(0) : 0;
    sqlite3_reset(sel.raw());  // release the read snapshot immediately, see below
    if (existing != 0) return existing;

    Stmt& ins = insert_term();
    ins.bind(1, term);
    ins.exec();
    return static_cast<int>(sqlite3_last_insert_rowid(db_));
}

void SqliteTextIndex::replace_record_terms(
    const std::string& table, int id,
    const std::unordered_map<int, int>& term_counts) {
    validate_table(table);
    {
        Stmt& del = delete_terms_stmt(table);
        del.bind(1, id);
        del.exec();
    }
    for (const auto& [term_id, count] : term_counts) {
        Stmt& ins = insert_terms_stmt(table);
        ins.bind(1, id).bind(2, term_id).bind(3, count);
        ins.exec();
    }
}

void SqliteTextIndex::set_counts(const std::string& table, int id,
                                 int unigram_count, int bigram_count) {
    validate_table(table);
    Stmt& upd = update_counts_stmt(table);
    upd.bind(1, unigram_count).bind(2, bigram_count).bind(3, id);
    upd.exec();
}

int SqliteTextIndex::corpus_size(const std::string& table) {
    validate_table(table);
    Stmt& s = count_stmt(table);
    int result = s.step() ? s.column_int(0) : 0;
    sqlite3_reset(s.raw());  // release the read snapshot immediately, see upsert_term
    return result;
}

int SqliteTextIndex::doc_freq(const std::string& table, int term_id) {
    validate_table(table);
    Stmt& s = doc_freq_stmt(table);
    s.bind(1, term_id);
    int result = s.step() ? s.column_int(0) : 0;
    sqlite3_reset(s.raw());  // release the read snapshot immediately, see upsert_term
    return result;
}

int SqliteTextIndex::lookup_term(const std::string& term) {
    Stmt& s = select_term_id();
    s.bind(1, term);
    int result = s.step() ? s.column_int(0) : 0;
    sqlite3_reset(s.raw());  // release the read snapshot immediately, see upsert_term
    return result;
}

std::vector<TextIndex::Posting> SqliteTextIndex::postings(
    const std::string& table, int term_id) {
    validate_table(table);
    Stmt& s = postings_stmt(table);
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
        Stmt& c = count_stmt(table);
        if (c.step()) total = c.column_int(0);
    }

    // Two-pass batches: read a bounded batch (finalize the SELECT), then write.
    // Walking a SELECT while UPDATE-ing the same table on one connection can
    // invalidate the cursor mid-scan (see rebuild_phon's note).
    //
    // This SELECT's shape is loop-invariant (select_sql doesn't change across
    // outer iterations) but reindex_table() itself is a rare admin/CLI
    // operation, not a per-row hot path -- so it's hoisted above the loop for
    // clarity, not promoted to a cached instance member.
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
