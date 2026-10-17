/**
 * sqlite::TextIndex — SQLite implementation of the custom v0.16 terms index.
 *
 * Holds a NON-OWNING sqlite3* (the sqlite::Backend owns the handle). Implements
 * the TextIndex storage hooks with ragger::Stmt, and provides create_schema()
 * (terms + 6 junction tables) and a batched full-table reindex_table().
 *
 * Hot-path statement caching: the handful of queries hit per-term/per-record
 * during indexing and search are cached as std::optional<Stmt> instance
 * members (lazy: prepared on first use, sqlite3_reset+clear_bindings on
 * reuse), one accessor per table since the 5 text tables are a small fixed
 * set (turns, turn_summaries, summaries, documents, decisions) — no map, no
 * runtime SQL-string formatting for these. A small if/else dispatch
 * (doc_freq_stmt/count_stmt/postings_stmt/delete_terms_stmt/
 * insert_terms_stmt/update_counts_stmt) picks the right per-table accessor
 * for callers that only know the table name at runtime; the SQL text itself
 * always lives next to its own named accessor, not in the dispatcher.
 */
#pragma once

#include <sqlite3.h>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../text_index.h"
#include "sqlite/tables.h"
#include "util/sqlite.h"

namespace ragger::sqlite {

// The shared embed/index text formulas (kDecisionEmbedTextSQL /
// kDocEmbedTextSQL) now live in sqlite/tables.h next to the registry rows
// that reference them (PHASE3 Step 1). tables.h is included here, so every
// existing `#include "sqlite/text_index.h"` user still sees them.

class TextIndex final : public ragger::TextIndex {
public:
    explicit TextIndex(sqlite3* db) : db_(db) {}

    // CREATE terms + <t>_terms junction tables + term_id indexes (idempotent).
    void create_schema();

    // Junction DDL in ONE place (PHASE3 Step 2): emits the current 0.16
    // junction shape (rowid table, PRIMARY KEY (<pk>, term_id), term_id
    // index). Phase 9 changes the shape here and nowhere else.
    static std::string junction_ddl(const TextTable& t);
    static std::string junction_index_ddl(const TextTable& t);

    // Full rebuild of one text table's index. Key-paginated read (batch 200),
    // then per-record index_record(). Idempotent (delete-then-insert). Returns
    // the number of records reindexed. `progress` prints a counter to stdout.
    int reindex_table(const std::string& table, bool progress = false);

    // Truncate the shared `terms` table and reset its AUTOINCREMENT counter,
    // starting term_ids fresh from 1. Only safe when reindexing ALL six
    // text tables in the same run: every junction table's term_id column is
    // `REFERENCES terms(term_id) ON DELETE CASCADE`, so this one DELETE
    // cascades and empties every junction table too (foreign_keys must be ON,
    // which sqlite::Backend always sets at connection open). Calling this
    // before reindexing only SOME tables would orphan the other tables'
    // still-valid junction rows against a wiped terms list -- callers MUST
    // reindex every table in the same pass after calling this.
    void reset_terms_table();

protected:
    int  upsert_term(const std::string& term) override;
    void replace_record_terms(const std::string& table, int id,
        const std::unordered_map<int, int>& term_counts) override;
    void set_counts(const std::string& table, int id,
                    int unigram_count, int bigram_count) override;
    int  corpus_size(const std::string& table) override;
    int  doc_freq(const std::string& table, int term_id) override;
    int  lookup_term(const std::string& term) override;
    std::vector<Posting> postings(const std::string& table, int term_id) override;

private:
    // Throws std::runtime_error if `table` is not one of the 5 known text
    // tables — guards the string-concatenated SQL against injection.
    static void validate_table(const std::string& table);
    static std::string id_col_of(const std::string& table);
    // SELECT id_col, text FROM <table> ORDER BY id_col — combines the doc
    // title formula for documents; turns are handled in reindex_table().
    static std::string text_select_of(const std::string& table);

    // ---- cached-statement plumbing ----------------------------------------
    // If `slot` is empty, prepares `sql` into it; otherwise resets + clears
    // bindings so it's ready to bind+step again. Returns the live statement.
    // Must be called on an instance (never static/global) -- a prepared
    // statement is bound to the specific sqlite3* it was prepared against.
    Stmt& cached(std::optional<Stmt>& slot, const char* sql);

    // Shared across upsert_term()'s SELECT and lookup_term() -- same SQL.
    Stmt& select_term_id();
    Stmt& insert_term();

    // Registry-driven statement cache (PHASE3 Step 2, B §2.2): one set of
    // six cached statements per registry row (sqlite/tables.h), generated
    // from ONE std::format template per verb. The SQL text is byte-identical
    // (whitespace-normalized) to the old 36 hand-written accessors — pinned
    // by the Step 2 identity probe (REPORT-P3).
    struct TableStmts {
        std::optional<Stmt> doc_freq, count, postings,
                            delete_terms, insert_terms, update_counts;
    };

    // Runtime dispatch (table name -> registry row -> cached statement).
    // Throws std::runtime_error("Invalid table name: ...") for unknowns.
    Stmt& doc_freq_stmt(const std::string& table);
    Stmt& count_stmt(const std::string& table);
    Stmt& postings_stmt(const std::string& table);
    Stmt& delete_terms_stmt(const std::string& table);
    Stmt& insert_terms_stmt(const std::string& table);
    Stmt& update_counts_stmt(const std::string& table);

    sqlite3* db_;   // non-owning; lifetime managed by sqlite::Backend

    // db_ is declared above these members so it outlives them: C++ destroys
    // members in reverse declaration order, so every cached Stmt finalizes
    // itself (via std::optional<Stmt>'s destructor) before db_ would ever be
    // closed by the owner.
    std::optional<Stmt> select_term_id_;
    std::optional<Stmt> insert_term_;

    std::array<TableStmts, kTextTableCount> stmts_;
};

}  // namespace ragger::sqlite
