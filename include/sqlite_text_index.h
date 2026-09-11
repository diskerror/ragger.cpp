/**
 * SqliteTextIndex — SQLite implementation of the custom v0.16 terms index.
 *
 * Holds a NON-OWNING sqlite3* (the SqliteBackend owns the handle). Implements
 * the TextIndex storage hooks with ragger::Stmt, and provides create_schema()
 * (terms + 5 junction tables) and a batched full-table reindex_table().
 */
#pragma once

#include <sqlite3.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "text_index.h"

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

    sqlite3* db_;   // non-owning; lifetime managed by SqliteBackend
};

}  // namespace ragger
