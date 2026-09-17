/**
 * SqliteTextIndex — SQLite implementation of the custom v0.16 terms index.
 *
 * Holds a NON-OWNING sqlite3* (the SqliteBackend owns the handle). Implements
 * the TextIndex storage hooks with ragger::Stmt, and provides create_schema()
 * (terms + 5 junction tables) and a batched full-table reindex_table().
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

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "text_index.h"
#include "util/sqlite.h"

namespace ragger {

class SqliteTextIndex final : public TextIndex {
public:
    explicit SqliteTextIndex(sqlite3* db) : db_(db) {}

    // CREATE terms + <t>_terms junction tables + term_id indexes (idempotent).
    void create_schema();

    // Full rebuild of one text table's index. Key-paginated read (batch 200),
    // then per-record index_record(). Idempotent (delete-then-insert). Returns
    // the number of records reindexed. `progress` prints a counter to stdout.
    int reindex_table(const std::string& table, bool progress = false);

    // Truncate the shared `terms` table and reset its AUTOINCREMENT counter,
    // starting term_ids fresh from 1. Only safe when reindexing ALL five
    // text tables in the same run: every junction table's term_id column is
    // `REFERENCES terms(term_id) ON DELETE CASCADE`, so this one DELETE
    // cascades and empties every junction table too (foreign_keys must be ON,
    // which SqliteBackend always sets at connection open). Calling this
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

    // Per-table accessors (fixed 5-table set: turns, turn_summaries,
    // summaries, documents, decisions). Named verb_noun_table so related
    // statements alphabetize/group together.
    Stmt& doc_freq_turns();
    Stmt& doc_freq_turn_summaries();
    Stmt& doc_freq_summaries();
    Stmt& doc_freq_documents();
    Stmt& doc_freq_decisions();

    Stmt& count_turns();
    Stmt& count_turn_summaries();
    Stmt& count_summaries();
    Stmt& count_documents();
    Stmt& count_decisions();

    Stmt& postings_turns();
    Stmt& postings_turn_summaries();
    Stmt& postings_summaries();
    Stmt& postings_documents();
    Stmt& postings_decisions();

    Stmt& delete_terms_turns();
    Stmt& delete_terms_turn_summaries();
    Stmt& delete_terms_summaries();
    Stmt& delete_terms_documents();
    Stmt& delete_terms_decisions();

    Stmt& insert_terms_turns();
    Stmt& insert_terms_turn_summaries();
    Stmt& insert_terms_summaries();
    Stmt& insert_terms_documents();
    Stmt& insert_terms_decisions();

    Stmt& update_counts_turns();
    Stmt& update_counts_turn_summaries();
    Stmt& update_counts_summaries();
    Stmt& update_counts_documents();
    Stmt& update_counts_decisions();

    // Runtime dispatch to the per-table accessor above, for the callers
    // (corpus_size/doc_freq/postings/replace_record_terms/set_counts) that
    // only know `table` as a validated string at call time. Plain if/else
    // over the fixed 5-table set -- no map, no format-string SQL.
    Stmt& doc_freq_stmt(const std::string& table);
    Stmt& count_stmt(const std::string& table);
    Stmt& postings_stmt(const std::string& table);
    Stmt& delete_terms_stmt(const std::string& table);
    Stmt& insert_terms_stmt(const std::string& table);
    Stmt& update_counts_stmt(const std::string& table);

    sqlite3* db_;   // non-owning; lifetime managed by SqliteBackend

    // db_ is declared above these members so it outlives them: C++ destroys
    // members in reverse declaration order, so every cached Stmt finalizes
    // itself (via std::optional<Stmt>'s destructor) before db_ would ever be
    // closed by the owner.
    std::optional<Stmt> select_term_id_;
    std::optional<Stmt> insert_term_;

    std::optional<Stmt> doc_freq_turns_, doc_freq_turn_summaries_,
                         doc_freq_summaries_, doc_freq_documents_,
                         doc_freq_decisions_;

    std::optional<Stmt> count_turns_, count_turn_summaries_, count_summaries_,
                         count_documents_, count_decisions_;

    std::optional<Stmt> postings_turns_, postings_turn_summaries_,
                         postings_summaries_, postings_documents_,
                         postings_decisions_;

    std::optional<Stmt> delete_terms_turns_, delete_terms_turn_summaries_,
                         delete_terms_summaries_, delete_terms_documents_,
                         delete_terms_decisions_;

    std::optional<Stmt> insert_terms_turns_, insert_terms_turn_summaries_,
                         insert_terms_summaries_, insert_terms_documents_,
                         insert_terms_decisions_;

    std::optional<Stmt> update_counts_turns_, update_counts_turn_summaries_,
                         update_counts_summaries_, update_counts_documents_,
                         update_counts_decisions_;
};

}  // namespace ragger
