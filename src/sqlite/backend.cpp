/**
 * SQLite backend for Ragger Memory (C++ port)
 */
#include "sqlite/backend.h"
#include "backend_impl.h"
#include "config.h"
#include "util/time.h"
#include "nlohmann_json.hpp"

#include <sqlite3.h>

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ragger::sqlite {

using json = nlohmann::json;


// -----------------------------------------------------------------------
// Public API (delegates to Impl)
// -----------------------------------------------------------------------
SqliteBackend::SqliteBackend(Embedder& embedder, const std::string& db_path)
    : pImpl(std::make_unique<Impl>(embedder, db_path)) {}

SqliteBackend::SqliteBackend(const std::string& db_path, bool readonly)
    : pImpl(std::make_unique<Impl>(db_path, readonly)) {}

SqliteBackend::~SqliteBackend() = default;

std::string SqliteBackend::db_path() const { return pImpl->db_path; }

std::string SqliteBackend::store(const std::string& text, json metadata, bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store(text, std::move(metadata), defer_embedding);
}

int SqliteBackend::store_document(const DocumentChunk& chunk, bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_document(chunk, defer_embedding);
}

int SqliteBackend::store_turn(const std::string& user_text,
                              const std::string& assistant_text,
                              const std::string& model_name, bool defer_embedding,
                              const std::string& session_guid,
                              const std::string& source_timestamp,
                              const std::string& session_name,
                              const std::string& name_source) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_turn(user_text, assistant_text, model_name,
                             defer_embedding, session_guid, source_timestamp,
                             session_name, name_source);
}

std::vector<TurnRecord> SqliteBackend::turns_by_session(
        const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turns_by_session(session_guid);
}

bool SqliteBackend::finalize_turn(int turn_id, const std::string& assistant_text,
                                  const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->finalize_turn(turn_id, assistant_text, model_name);
}

int SqliteBackend::store_summary(const std::string& text, const std::string& level,
                                 const std::string& model_name,
                                 const std::string& session_guid,
                                 const std::string& source_timestamp,
                                 const std::string& tags) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_summary(text, level, model_name, session_guid,
                                source_timestamp, tags);
}

bool SqliteBackend::summary_exists_exact(const std::string& text, const std::string& created_at) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->summary_exists_exact(text, created_at);
}

bool SqliteBackend::decision_exists_exact(const std::string& text, const std::string& created_at) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->decision_exists_exact(text, created_at);
}

bool SqliteBackend::turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                                      int window_seconds) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_exists_fuzzy(user_text, ts, window_seconds);
}

std::optional<TurnRecord> SqliteBackend::find_turn_by_text(const std::string& user_text) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->find_turn_by_text(user_text);
}

bool SqliteBackend::update_turn_meta(int turn_id, const std::string& timestamp,
                                     const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_turn_meta(turn_id, timestamp, session_guid);
}

bool SqliteBackend::update_summary_text(int summary_id, const std::string& text,
                                        const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_summary_text(summary_id, text, model_name);
}

bool SqliteBackend::turn_summary_exists(int turn_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_summary_exists(turn_id);
}

bool SqliteBackend::finalize_turn_summary(int turn_id, const std::string& text,
                                          const std::string& summary_model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->finalize_turn_summary(turn_id, text, summary_model_name);
}

bool SqliteBackend::mark_turn_summarized(int turn_id, const std::string& model_name) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->mark_turn_summarized(turn_id, model_name);
}

int SqliteBackend::reset_abandoned_turn_summaries(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->reset_abandoned_turn_summaries(limit);
}

bool SqliteBackend::set_summary_tags(int summary_id, const std::string& tags) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_summary_tags(summary_id, tags);
}

std::vector<std::string> SqliteBackend::recent_summaries(const std::string& level,
                                                         int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->recent_summaries(level, limit);
}

std::vector<std::string> SqliteBackend::current_decisions(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->current_decisions(limit);
}

bool SqliteBackend::set_decision_status(int decision_id, const std::string& status) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_decision_status(decision_id, status);
}

std::vector<std::string> SqliteBackend::decisions_by_status(const std::string& status, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->decisions_by_status(status, limit);
}

std::vector<TurnRecord> SqliteBackend::unsummarized_turns(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->unsummarized_turns(limit);
}

std::vector<DraftSummary> SqliteBackend::draft_summaries(int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->draft_summaries(limit);
}

std::vector<std::string> SqliteBackend::sessions_needing_close(int pause_minutes) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->sessions_needing_close(pause_minutes);
}

std::string SqliteBackend::last_episode_end(const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->last_episode_end(session_guid);
}

std::vector<SummaryRecord> SqliteBackend::l2_summaries_since(
        const std::string& session_guid, const std::string& since_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->l2_summaries_since(session_guid, since_ts);
}

std::vector<EpisodeCandidateTurn> SqliteBackend::episode_candidate_turns(
        const std::string& session_guid, const std::string& since_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->episode_candidate_turns(session_guid, since_ts);
}

int SqliteBackend::store_episode(const std::string& text,
        const std::string& model_name, const std::string& session_guid,
        const std::string& first_ts, const std::string& last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_episode(text, model_name, session_guid, first_ts, last_ts);
}

std::vector<std::string> SqliteBackend::episodes_needing_close(int idle_minutes) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->episodes_needing_close(idle_minutes);
}

std::vector<std::string> SqliteBackend::episode_texts(const std::string& session_guid) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->episode_texts(session_guid);
}

bool SqliteBackend::set_summary_updated_at(int summary_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->set_summary_updated_at(summary_id);
}

std::vector<ClosedRun> SqliteBackend::sessions_needing_close_boundary() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->sessions_needing_close_boundary();
}

std::vector<ClosedRun> SqliteBackend::projects_needing_close_boundary(int gap_days) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->projects_needing_close_boundary(gap_days);
}

void SqliteBackend::advance_session_boundary_watermark(int turn_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->advance_session_boundary_watermark(turn_id);
}

void SqliteBackend::advance_project_boundary_watermark(int turn_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->advance_project_boundary_watermark(turn_id);
}

std::vector<std::string> SqliteBackend::bounded_session_rollup_texts(
        const std::string& session_guid, int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->bounded_session_rollup_texts(session_guid, first_ts, last_ts);
}

std::vector<std::string> SqliteBackend::bounded_project_rollup_texts(
        int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->bounded_project_rollup_texts(first_ts, last_ts);
}

int SqliteBackend::store_session_summary(const std::string& text,
        const std::string& model_name, const std::string& session_guid,
        int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_session_summary(text, model_name, session_guid, first_ts, last_ts);
}

int SqliteBackend::store_project_summary(const std::string& text,
        const std::string& model_name, int64_t first_ts, int64_t last_ts) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_project_summary(text, model_name, first_ts, last_ts);
}

std::vector<TurnRecord> SqliteBackend::turns_by_session_desc(
        const std::string& session_guid, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turns_by_session_desc(session_guid, limit);
}

std::vector<SummaryRecord>
SqliteBackend::turn_summaries_by_session_desc(
        const std::string& session_guid, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->turn_summaries_by_session_desc(session_guid, limit);
}

std::vector<SummaryRecord>
SqliteBackend::session_summaries_desc(
        const std::string& session_guid, int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->session_summaries_desc(session_guid, limit);
}

bool SqliteBackend::update_text(int memory_id, const std::string& text, json metadata, bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_text(memory_id, text, std::move(metadata), defer_embedding);
}

SearchResponse SqliteBackend::search(const std::string& query, int limit,
                                     float min_score,
                                     std::vector<std::string> collections) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->search(query, limit, min_score, std::move(collections));
}

SearchResponse SqliteBackend::search_text_only(const std::string& query,
                                               int limit) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->search_text_only(query, limit);
}

int SqliteBackend::count() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->count(); }

std::vector<std::pair<std::string, int64_t>> SqliteBackend::table_row_counts() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->table_row_counts(); }

bool SqliteBackend::has_embeddings() const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->has_embeddings(); }

int SqliteBackend::count_embeddable_rows(const std::string& table) const { std::lock_guard<std::mutex> lk(pImpl->mu); return pImpl->count_embeddable_rows(table); }

std::vector<SearchResult> SqliteBackend::load_all(const std::string& collection) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->load_all(collection);
}

int SqliteBackend::rebuild_embeddings(Embedder& embedder, bool progress, const std::string& table) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->rebuild_embeddings(embedder, progress, table);
}

int SqliteBackend::backfill_embeddings(Embedder& embedder, const std::string& table) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->backfill_embeddings(embedder, table);
}

uint8_t SqliteBackend::embedding_version() const {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->get_embedding_version();
}

uint8_t SqliteBackend::increment_embedding_version() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->increment_embedding_version();
}

int SqliteBackend::reindex_table(const std::string& table, bool progress) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->reindex_table(table, progress);
}

void SqliteBackend::reset_terms_table() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->reset_terms_table();
}

bool SqliteBackend::update_document_embedding(int document_id,
                                              const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_document_embedding(document_id, emb);
}

int SqliteBackend::store_decision(const std::string& text,
                                  const std::string& status,
                                  const std::string& tags,
                                  const std::string& source_timestamp,
                                  bool defer_embedding) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->store_decision(text, status, tags, source_timestamp, defer_embedding);
}

bool SqliteBackend::update_decision_embedding(int decision_id,
                                              const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_decision_embedding(decision_id, emb);
}

bool SqliteBackend::update_summary_embedding(int summary_id,
                                             const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_summary_embedding(summary_id, emb);
}

bool SqliteBackend::update_turn_embedding(int turn_id,
                                          const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->update_turn_embedding(turn_id, emb);
}

std::vector<std::string> SqliteBackend::collections() const {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->collections();
}

void SqliteBackend::close() { pImpl->close(); }

bool SqliteBackend::delete_memory(int memory_id) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->delete_memory(memory_id);
}

int SqliteBackend::delete_batch(const std::vector<int>& memory_ids) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->delete_batch(memory_ids);
}

std::vector<SearchResult> SqliteBackend::search_by_metadata(const json& metadata_filter, int limit,
                                                           const std::string& after,
                                                           const std::string& before) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->search_by_metadata(metadata_filter, limit, after, before);
}

int SqliteBackend::cleanup_old_conversations(float max_age_hours) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    auto cutoff = std::chrono::system_clock::now() -
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::duration<double, std::ratio<3600>>(max_age_hours));
    int64_t cutoff_epoch = db_epoch(std::chrono::system_clock::to_time_t(cutoff));

    // v2: raw verbatim exchanges (L1 turns) are what age out by retention;
    // their gist is preserved in the L2/L3 summaries. Purge old turns by
    // timestamp. (Pre-v2 this deleted summaries tagged collection='conversation';
    // that column is gone in the lean schema.)
    Stmt stmt(pImpl->db,
        "DELETE FROM turns WHERE created_at < ?");
    stmt.bind(1, cutoff_epoch);

    int deleted = 0;
    if (stmt.exec()) {
        deleted = static_cast<int>(sqlite3_changes(pImpl->db));
    }
    return deleted;
}

// ---- users / settings CRUD (delegate through mutex) -----------------------

std::optional<UserInfo> SqliteBackend::get_user_by_username(const std::string& username) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->get_user_by_username(username);
}

std::optional<std::string> SqliteBackend::get_user_password(const std::string& username) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->get_user_password(username);
}

void SqliteBackend::update_user_token(const std::string& username, const std::string& new_hash) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->update_user_token(username, new_hash);
}

int SqliteBackend::create_user(const std::string& username, const std::string& token_hash) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->create_user(username, token_hash);
}

bool SqliteBackend::delete_user(const std::string& username) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->delete_user(username);
}

void SqliteBackend::set_user_password(const std::string& username,
                                      const std::string& password_hash) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->set_user_password(username, password_hash);
}

std::optional<UserInfo> SqliteBackend::get_user_by_token_hash(const std::string& token_hash) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->get_user_by_token_hash(token_hash);
}

std::optional<std::string> SqliteBackend::get_setting(const std::string& key) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->get_setting(key);
}

void SqliteBackend::set_setting(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->set_setting(key, value);
}

// ---- schema introspection (delegate through mutex) ------------------------

std::vector<SchemaObject> SqliteBackend::list_schema_objects() {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->list_schema_objects();
}

std::vector<std::string> SqliteBackend::table_column_names(const std::string& table) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    return pImpl->table_column_names(table);
}

void SqliteBackend::iterate_table_rows(const std::string& table,
                                       const std::function<void(const ExportRow&)>& cb) {
    std::lock_guard<std::mutex> lk(pImpl->mu);
    pImpl->iterate_table_rows(table, cb);
}

} // namespace ragger::sqlite
