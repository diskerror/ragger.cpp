/**
 * RaggerMemory - high-level facade implementation
 */

#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "diskerror/logger.h"
#include "ragger/lang.h"
#include <format>

namespace ragger {

RaggerMemory::RaggerMemory(const std::string& db_path,
                           const std::string& model_dir,
                           const std::string& user_db_path,
                           bool skip_embedding_guard)
{
    // Resolve model directory from config or override
    std::string resolved_model_dir = model_dir.empty()
        ? config().resolved_model_dir()
        : expand_path(model_dir);

    // Create embedder (shared across backends)
    embedder_ = std::make_unique<Embedder>(resolved_model_dir);

    // Create primary backend - single user DB only now
    backend_ = std::make_unique<SqliteBackend>(*embedder_, db_path);
    
    // --- Embedding identity drift guard (model + dtype + dimensions) -------
    // The settings table records what built this DB. On startup we compare it
    // to the current config; thereafter the stored values are authoritative.
    //   - first use (no stored value)        → record it
    //   - match                              → fine
    //   - mismatch + DB has embeddings       → STOP (rebuild or revert the value)
    //   - mismatch + empty DB                → re-adopt the new value (nothing to
    //                                          invalidate yet)
    //   - skip_embedding_guard (rebuild)     → don't touch settings; the
    //                                          rebuild path rewrites them after
    //                                          re-encoding (so an aborted rebuild
    //                                          leaves settings≠config and the
    //                                          guard still catches it next time).
    const std::string current_model = config().resolve_model(config().embedding_model);
    const std::string current_vtype =
        (config().embedding_vector_type == "f32") ? "f32" : "f16";
    const std::string current_dims = std::to_string(config().embedding_dimensions);

    if (!skip_embedding_guard) {
        const bool has_data = backend_->has_embeddings();
        auto guard = [&](const char* key, const std::string& current,
                         const std::string& err) {
            auto stored = backend_->get_setting(key);
            if (!stored.has_value() || *stored == current) {
                if (!stored.has_value()) backend_->set_setting(key, current);
                return;
            }
            if (has_data) throw std::runtime_error(err);
            backend_->set_setting(key, current);  // empty DB: re-adopt
        };
        guard("embedding_model", current_model,
              std::format(lang::ERR_EMBEDDING_MISMATCH,
                          backend_->get_setting("embedding_model").value_or(""), current_model));
        guard("vector_type", current_vtype,
              std::format(lang::ERR_VECTOR_TYPE_MISMATCH,
                          backend_->get_setting("vector_type").value_or(""), current_vtype));
        guard("dimensions", current_dims,
              std::format(lang::ERR_DIMENSIONS_MISMATCH,
                          backend_->get_setting("dimensions").value_or(""), current_dims));
    }

    // Backfill any rows left without embeddings (deferred-embedding writes
    // that didn't get processed before the previous shutdown). The query is
    // a no-op when nothing is NULL; embedder is already loaded above.
    int filled = backend_->backfill_embeddings(*embedder_);
    if (filled > 0) {
        Diskerror::logger::info(std::format(lang::MSG_BACKFILLED_EMBEDDINGS, filled));
    }
}

RaggerMemory::~RaggerMemory() {
    close();
}

std::string RaggerMemory::store(const std::string& text, json metadata,
                                 bool common, bool defer_embedding) {
    // common flag is now ignored - single-user mode only
    return backend_->store(text, std::move(metadata), defer_embedding);
}

int RaggerMemory::store_document(const DocumentChunk& chunk, bool defer_embedding) {
    return backend_->store_document(chunk, defer_embedding);
}

int RaggerMemory::store_turn(const std::string& user_text,
                             const std::string& assistant_text,
                             const std::string& model_name, bool defer_embedding,
                             const std::string& session_guid) {
    return backend_->store_turn(user_text, assistant_text, model_name,
                                defer_embedding, session_guid);
}

bool RaggerMemory::finalize_turn(int turn_id, const std::string& assistant_text,
                                 const std::string& model_name) {
    return backend_->finalize_turn(turn_id, assistant_text, model_name);
}

bool RaggerMemory::update_document_embedding(int document_id,
                                             const std::vector<float>& emb) {
    return backend_->update_document_embedding(document_id, emb);
}

int RaggerMemory::store_summary(const std::string& text, const std::string& level,
                                const std::string& status, const std::string& model_name,
                                const std::string& session_guid) {
    return backend_->store_summary(text, level, status, model_name, session_guid);
}

std::optional<std::pair<int, std::string>>
RaggerMemory::current_session_summary(const std::string& session_guid) {
    return backend_->current_session_summary(session_guid);
}

bool RaggerMemory::update_summary_text(int summary_id, const std::string& text,
                                       const std::string& model_name) {
    return backend_->update_summary_text(summary_id, text, model_name);
}

bool RaggerMemory::set_summary_status(int summary_id, const std::string& status) {
    return backend_->set_summary_status(summary_id, status);
}

std::vector<std::string> RaggerMemory::recent_summaries(const std::string& level,
                                                        int limit) {
    return backend_->recent_summaries(level, limit);
}

std::vector<std::string> RaggerMemory::current_decisions(int limit) {
    return backend_->current_decisions(limit);
}

bool RaggerMemory::update_text(int memory_id, const std::string& text, json metadata,
                                bool defer_embedding) {
    return backend_->update_text(memory_id, text, std::move(metadata), defer_embedding);
}

SearchResponse RaggerMemory::search(const std::string& query,
                                    int limit,
                                    float min_score,
                                    std::vector<std::string> collections) {
    return backend_->search(query, limit, min_score, std::move(collections));
}

int RaggerMemory::count() const {
    return backend_->count();
}

std::vector<SearchResult> RaggerMemory::load_all(const std::string& collection) {
    return backend_->load_all(collection);
}

int RaggerMemory::rebuild_embeddings() {
    return backend_->rebuild_embeddings(*embedder_);
}

int RaggerMemory::backfill_embeddings() {
    return backend_->backfill_embeddings(*embedder_);
}

std::vector<std::string> RaggerMemory::collections() const {
    return backend_->collections();
}

bool RaggerMemory::delete_memory(int memory_id) {
    return backend_->delete_memory(memory_id);
}

int RaggerMemory::delete_batch(const std::vector<int>& memory_ids) {
    return backend_->delete_batch(memory_ids);
}

std::vector<SearchResult> RaggerMemory::search_by_metadata(const json& metadata_filter, int limit,
                                                           const std::string& after,
                                                           const std::string& before) {
    return backend_->search_by_metadata(metadata_filter, limit, after, before);
}

void RaggerMemory::close() {
    if (backend_) backend_->close();
}

CaptureResult capture_turn(RaggerMemory& memory,
                           const std::string& user,
                           const std::string& assistant,
                           const std::string& model,
                           const std::string& session_id) {
    // Gate: turn capture is opt-in. Agent-driven store/search are unaffected.
    if (!config().capture_turns) return {false, -1};
    // A turn needs at least the user side; skip empty pushes.
    if (user.empty()) return {false, -1};
    int turn_id = memory.store_turn(user, assistant, model,
                                    /*defer_embedding=*/false, session_id);
    return {true, turn_id};
}

SessionContext build_context(RaggerMemory& memory, const std::string& session_id) {
    // Gate: the read side requires both capture (something to read) and the
    // build-context toggle. Off → enabled=false, no turns.
    if (!config().capture_turns || !config().build_context) return {false, {}};
    if (session_id.empty()) return {true, {}};
    return {true, memory.backend()->turns_by_session(session_id)};
}

} // namespace ragger
