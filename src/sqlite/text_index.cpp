/**
 * sqlite_text_index.cpp — SQLite storage hooks for the custom terms index.
 */
#include "sqlite/text_index.h"

#include <format>
#include <iostream>
#include <stdexcept>

#include "util/sqlite.h"

namespace ragger::sqlite {

namespace {

// Registry row for a validated table name; throws the same message the old
// hand lists threw (B §2.2: one source of truth, sqlite/tables.h).
const TextTable& table_or_throw(const std::string& table) {
    const TextTable* t = find_text_table(table);
    if (!t) throw std::runtime_error("Invalid table name: " + table);
    return *t;
}

// One std::format template per verb (PHASE3 Step 2). The generated text is
// byte-identical, whitespace-normalized, to the 36 old hand-written
// accessors — verified by the Step 2 identity probe (REPORT-P3).
std::string sql_doc_freq(const TextTable& t) {
    return std::format("SELECT COUNT(*) FROM {} WHERE term_id = ?", t.junction);
}
std::string sql_count(const TextTable& t) {
    return std::format("SELECT COUNT(*) FROM {}", t.table);
}
std::string sql_postings(const TextTable& t) {
    return std::format(
        "SELECT j.{0}, j.count, b.unigram_count, b.bigram_count "
        "FROM {1} j JOIN {2} b ON b.{0} = j.{0} "
        "WHERE j.term_id = ?",
        t.pk, t.junction, t.table);
}
std::string sql_delete_terms(const TextTable& t) {
    return std::format("DELETE FROM {} WHERE {} = ?", t.junction, t.pk);
}
std::string sql_insert_terms(const TextTable& t) {
    return std::format(
        "INSERT OR REPLACE INTO {} ({}, term_id, count) VALUES (?, ?, ?)",
        t.junction, t.pk);
}
std::string sql_update_counts(const TextTable& t) {
    return std::format(
        "UPDATE {} SET unigram_count = ?, bigram_count = ? WHERE {} = ?",
        t.table, t.pk);
}

} // namespace

void TextIndex::validate_table(const std::string& table) {
    table_or_throw(table);
}

std::string TextIndex::id_col_of(const std::string& table) {
    return table_or_throw(table).pk;
}

std::string TextIndex::text_select_of(const std::string& table) {
    const TextTable& t = table_or_throw(table);
    if (table == "turns")  // user_text + assistant_text special case
        return "SELECT turn_id, user_text, assistant_text FROM turns ORDER BY turn_id";
    if (t.embed_sql)
        return std::format("SELECT {0}, {1} AS text FROM {2} ORDER BY {0}",
                           t.pk, t.embed_sql, t.table);
    return std::format("SELECT {}, text FROM {} ORDER BY {}", t.pk, t.table, t.pk);
}

void TextIndex::create_schema() {
    auto exec = [&](const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "unknown error";
            sqlite3_free(err);
            throw std::runtime_error("TextIndex schema: " + msg);
        }
    };

    exec(R"(
        CREATE TABLE IF NOT EXISTS terms (
            term_id INTEGER PRIMARY KEY AUTOINCREMENT,
            term    TEXT NOT NULL UNIQUE
        )
    )");

    // One junction table per registry row. FK ON DELETE CASCADE means a
    // deleted record's index rows go away automatically — nothing to
    // maintain in the write path beyond delete-then-insert for updates.
    for (const auto& t : kTextTables) {
        exec(junction_ddl(t));
        exec(junction_index_ddl(t));
    }
}

std::string TextIndex::junction_ddl(const TextTable& t) {
    // PHASE9: final shape — WITHOUT ROWID clustered on (term_id, <pk>).
    // The read-hot path (score_query: term -> postings with count) is one
    // contiguous range of the PK B-tree; the write path (delete-then-insert
    // by <pk>) uses the secondary idx_<table>_terms_row index. One copy per
    // posting instead of the 0.16 triple (rowid table + PK + term index).
    return std::format(
        "CREATE TABLE IF NOT EXISTS {} (\n"
        "    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,\n"
        "    {} INTEGER NOT NULL REFERENCES {}({}) ON DELETE CASCADE,\n"
        "    count   INTEGER NOT NULL DEFAULT 1,\n"
        "    PRIMARY KEY (term_id, {})\n"
        ") WITHOUT ROWID",
        t.junction, t.pk, t.table, t.pk, t.pk);
}

std::string TextIndex::junction_index_ddl(const TextTable& t) {
    return std::format("CREATE INDEX IF NOT EXISTS idx_{}_terms_row ON {}({})",
                       t.table, t.junction, t.pk);
}

// ---- cached-statement plumbing --------------------------------------------

Stmt& TextIndex::cached(std::optional<Stmt>& slot, const char* sql) {
    if (!slot) {
        slot.emplace(db_, sql);
    } else {
        sqlite3_reset(slot->raw());
        sqlite3_clear_bindings(slot->raw());
    }
    return *slot;
}

Stmt& TextIndex::select_term_id() {
    return cached(select_term_id_, "SELECT term_id FROM terms WHERE term = ?");
}
Stmt& TextIndex::insert_term() {
    return cached(insert_term_, "INSERT OR IGNORE INTO terms(term) VALUES (?)");
}

// ---- registry-driven cached statements -------------------------------------
// One dispatcher per verb: table name -> registry row -> generated SQL ->
// cached Stmt (PHASE3 Step 2). The generated text is byte-identical,
// whitespace-normalized, to the 36 old hand-written accessors (identity
// probe kept in REPORT-P3).

namespace {
size_t row_of(const std::string& table) {
    const TextTable& t = table_or_throw(table);
    return static_cast<size_t>(&t - kTextTables);
}
} // namespace

Stmt& TextIndex::doc_freq_stmt(const std::string& table) {
    const size_t i = row_of(table);
    std::string sql = sql_doc_freq(kTextTables[i]);
    return cached(stmts_[i].doc_freq, sql.c_str());
}
Stmt& TextIndex::count_stmt(const std::string& table) {
    const size_t i = row_of(table);
    std::string sql = sql_count(kTextTables[i]);
    return cached(stmts_[i].count, sql.c_str());
}
Stmt& TextIndex::postings_stmt(const std::string& table) {
    const size_t i = row_of(table);
    std::string sql = sql_postings(kTextTables[i]);
    return cached(stmts_[i].postings, sql.c_str());
}
Stmt& TextIndex::delete_terms_stmt(const std::string& table) {
    const size_t i = row_of(table);
    std::string sql = sql_delete_terms(kTextTables[i]);
    return cached(stmts_[i].delete_terms, sql.c_str());
}
Stmt& TextIndex::insert_terms_stmt(const std::string& table) {
    const size_t i = row_of(table);
    std::string sql = sql_insert_terms(kTextTables[i]);
    return cached(stmts_[i].insert_terms, sql.c_str());
}
Stmt& TextIndex::update_counts_stmt(const std::string& table) {
    const size_t i = row_of(table);
    std::string sql = sql_update_counts(kTextTables[i]);
    return cached(stmts_[i].update_counts, sql.c_str());
}

// ---- TextIndex overrides ---------------------------------------------------

void TextIndex::reset_terms_table() {
    // Drop-and-recreate instead of DELETE: on ~900K+ rows, DELETE FROM terms
    // has to walk every row, update the UNIQUE index B-tree, and write a WAL
    // frame per row -- O(n) work. DROP TABLE is a catalog-only operation
    // (SQLite just forgets the table/index exist), so it's near-instant
    // regardless of row count; CREATE TABLE then reuses create_schema()'s
    // exact DDL, so the two paths can never drift apart.
    //
    // Wrapped in one transaction: a crash mid-sequence rolls back to the
    // pre-reset state instead of leaving tables half-gone. When the caller
    // is already inside a transaction (the 0.16->0.17 leg runs this as its
    // last data step), JOIN it: no BEGIN/COMMIT of our own.
    //
    // No manual sqlite_sequence cleanup needed -- a freshly CREATE'd
    // AUTOINCREMENT table has no sqlite_sequence row until its first insert.
    //
    // Any cached Stmt bound to `terms` or a `<table>_terms` junction (e.g.
    // select_term_id_, insert_term_, the per-table junction insert/delete
    // members) is prepared via sqlite3_prepare_v2, which SQLite transparently
    // reprepares against the new schema on its next step() -- no code
    // elsewhere needs to change for this.
    char* err = nullptr;
    auto exec = [&](const char* sql) {
        if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "unknown error";
            sqlite3_free(err);
            throw std::runtime_error("TextIndex reset_terms_table: " + msg);
        }
    };
    const bool own_txn = sqlite3_get_autocommit(db_) != 0;
    if (own_txn)
        exec("BEGIN IMMEDIATE");
    try {
        // Junction tables reference terms(term_id); drop them first so no
        // statement can observe a junction row pointing at a gone table
        // (though within one transaction this is moot for correctness, it
        // keeps drop order mirroring the FK dependency direction).
        for (const auto& t : kTextTables)
            exec(("DROP TABLE IF EXISTS " + std::string(t.junction)).c_str());
        exec("DROP TABLE IF EXISTS terms");
        create_schema();  // CREATE TABLE/INDEX IF NOT EXISTS -- recreates all, empty
        if (own_txn)
            exec("COMMIT");
    } catch (...) {
        if (own_txn)
            exec("ROLLBACK");
        throw;
    }
}

int TextIndex::upsert_term(const std::string& term) {
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
    // A failed insert must not pass silently: last_insert_rowid would then
    // return a stale id and every junction row built from it is wrong.
    if (!ins.exec())
        throw std::runtime_error(std::format("TextIndex insert_term (terms, term {}): {}",
                                             term, sqlite3_errmsg(db_)));
    return static_cast<int>(sqlite3_last_insert_rowid(db_));
}

void TextIndex::replace_record_terms(
    const std::string& table, int id,
    const std::unordered_map<int, int>& term_counts) {
    validate_table(table);
    {
        Stmt& del = delete_terms_stmt(table);
        del.bind(1, id);
        if (!del.exec())
            throw std::runtime_error(std::format("TextIndex delete_terms ({}, id {}): {}",
                                                 table, id, sqlite3_errmsg(db_)));
    }
    for (const auto& [term_id, count] : term_counts) {
        Stmt& ins = insert_terms_stmt(table);
        ins.bind(1, id).bind(2, term_id).bind(3, count);
        // FK failures (id/term_id not present) used to fail silently here —
        // the exact mechanism that hid the turn_summaries mis-keying bug.
        if (!ins.exec())
            throw std::runtime_error(std::format("TextIndex insert_terms ({}, id {}): {}",
                                                 table, id, sqlite3_errmsg(db_)));
    }
}

void TextIndex::set_counts(const std::string& table, int id,
                                 int unigram_count, int bigram_count) {
    validate_table(table);
    Stmt& upd = update_counts_stmt(table);
    upd.bind(1, unigram_count).bind(2, bigram_count).bind(3, id);
    if (!upd.exec())
        throw std::runtime_error(std::format("TextIndex set_counts ({}, id {}): {}",
                                             table, id, sqlite3_errmsg(db_)));
}

int TextIndex::corpus_size(const std::string& table) {
    validate_table(table);
    Stmt& s = count_stmt(table);
    int result = s.step() ? s.column_int(0) : 0;
    sqlite3_reset(s.raw());  // release the read snapshot immediately, see upsert_term
    return result;
}

int TextIndex::doc_freq(const std::string& table, int term_id) {
    validate_table(table);
    Stmt& s = doc_freq_stmt(table);
    s.bind(1, term_id);
    int result = s.step() ? s.column_int(0) : 0;
    sqlite3_reset(s.raw());  // release the read snapshot immediately, see upsert_term
    return result;
}

int TextIndex::lookup_term(const std::string& term) {
    Stmt& s = select_term_id();
    s.bind(1, term);
    int result = s.step() ? s.column_int(0) : 0;
    sqlite3_reset(s.raw());  // release the read snapshot immediately, see upsert_term
    return result;
}

std::vector<TextIndex::Posting> TextIndex::postings(
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

int TextIndex::reindex_table(const std::string& table, bool progress) {
    validate_table(table);
    const std::string id_col = id_col_of(table);
    const std::string select_sql = text_select_of(table);

    int total = 0;
    if (progress) {
        Stmt& c = count_stmt(table);
        if (c.step()) total = c.column_int(0);
        c.reset();  // cached stmt: release the read lock before VACUUM etc.
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

}  // namespace ragger::sqlite
