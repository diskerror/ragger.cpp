/**
 * sqlite/tables.h — THE text-table registry (v0.17 rework, PHASE3 Step 1).
 *
 * One row per indexed/embedded text table. Before this, the table set was
 * hand-copied across ~11 lists (TextIndex schema/statements, embed_tables,
 * count_embeddable_rows, has_embeddings, junction views, known agent layers,
 * tests); every schema phase (4–9) would have had to touch each one. Now
 * each table is one row here, and every list above is derived.
 *
 * Row ORDER is embed_tables()'s old iteration order — the only one with
 * user-visible effects (rebuild progress prints follow it). DDL creation,
 * drops, and view creation are order-insensitive.
 *
 * `embed_sql` is the SQL expression producing the embedded/indexed text;
 * nullptr means the plain `text` column. `turns` is nullptr here but keeps
 * its special case everywhere it ever had one (user_text + assistant_text;
 * reindex_table and embed_tables handle it explicitly).
 *
 * Includes sqlite/text_index.h for the shared embed-text formulas — the
 * dependency direction is tables.h → text_index.h (text_index.h stays
 * self-contained, so there is no cycle).
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ragger::sqlite {

// ---- shared embed/index text formulas ---------------------------------------
// Single source of truth for the SQL column expressions that build the
// embedded/indexed text of the structured tables. Shared by
// TextIndex::text_select_of(), Backend::Impl::embed_tables() (maintenance),
// and the store paths that build the same string in C++: the text index, the
// embedding input, and the stored rows MUST agree, so these live here next
// to the registry and nowhere else (review 4B: they used to be public static
// members of Backend::Impl with a duplicated copy in text_index.cpp).
// Moved here from sqlite/text_index.h in PHASE3 Step 1: the registry rows
// reference them, and text_index.h includes tables.h (not the reverse), so
// there is no include cycle.

// decisions/insights embed/index text (db 0.17): `trigger + '\n' + text`
// when a trigger is present, else text alone (pre-0.17 rows). Trigger first
// on purpose: it is phrased in the future query's vocabulary (design doc
// §5.3a).
inline constexpr const char* kDecisionEmbedTextSQL =
    "CASE WHEN ifnull(trigger,'') = '' THEN text "
    "ELSE trigger || char(10) || text END";

// documents embed/index text: `text + '\n' + title` (title from the joined
// document_sources row), matching store_document so housekeeping rebuilds
// carry the same title signal the write path bakes in. char(10) == '\n'.
inline constexpr const char* kDocEmbedTextSQL =
    "text || CASE WHEN (SELECT ifnull(title,'') FROM document_sources ds "
    "WHERE ds.document_source_id = documents.document_source_id) = '' "
    "THEN '' ELSE char(10) || (SELECT title FROM document_sources ds "
    "WHERE ds.document_source_id = documents.document_source_id) END";

struct TextTable {
    const char* table;     // "turn_summaries"
    const char* pk;        // "turn_summary_id"
    const char* junction;  // "turn_summaries_terms"   (always <table>_terms)
    const char* layer;     // agent layer name ("turn_summary"); "" = not an agent layer
    const char* embed_sql; // SQL expression for the embedded/indexed text; nullptr = plain `text`
};

inline constexpr TextTable kTextTables[] = {
    {"turns",          "turn_id",         "turns_terms",          "turn",         nullptr},
    {"summaries",      "summary_id",      "summaries_terms",      "summary",      nullptr},
    {"decisions",      "decision_id",     "decisions_terms",      "decision",     kDecisionEmbedTextSQL},
    {"documents",      "document_id",     "documents_terms",      "document",     kDocEmbedTextSQL},
    {"turn_summaries", "turn_summary_id", "turn_summaries_terms", "turn_summary", nullptr},
    {"insights",       "insight_id",      "insights_terms",       "insight",      kDecisionEmbedTextSQL},
};

inline constexpr std::size_t kTextTableCount = std::size(kTextTables);

// nullptr when unknown (linear scan — the registry is tiny).
inline constexpr const TextTable* find_text_table(std::string_view name) {
    for (const auto& t : kTextTables)
        if (name == t.table) return &t;
    return nullptr;
}

// Comma-separated table names for usage/error text (registry-derived).
inline std::string text_table_names() {
    std::string out;
    for (const auto& t : kTextTables) {
        if (!out.empty()) out += ", ";
        out += t.table;
    }
    return out;
}

} // namespace ragger::sqlite
