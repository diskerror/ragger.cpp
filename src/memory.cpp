/**
 * RaggerMemory - high-level facade implementation
 */

#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "ragger/logs.h"
#include "ragger/lang.h"
#include <format>

namespace ragger {

RaggerMemory::RaggerMemory(const std::string& db_path,
                           const std::string& model_dir,
                           const std::string& user_db_path)
{
    // Resolve model directory from config or override
    std::string resolved_model_dir = model_dir.empty()
        ? config().resolved_model_dir()
        : expand_path(model_dir);

    // Create embedder (shared across backends)
    embedder_ = std::make_unique<Embedder>(resolved_model_dir);

    // Create primary backend - single user DB only now
    backend_ = std::make_unique<SqliteBackend>(*embedder_, db_path);
    
    // Model mismatch guard: stored embeddings are incompatible across models.
    const std::string current_model = config().resolve_model(config().embedding_model);
    auto stored_model = backend_->get_setting("embedding_model");
    if (!stored_model.has_value()) {
        // First use — record which model built this database.
        backend_->set_setting("embedding_model", current_model);
    } else if (*stored_model != current_model) {
        throw std::runtime_error(
            "Embedding model mismatch: database was built with '" + *stored_model +
            "' but config specifies '" + current_model + "'. " +
            "Reorganise your models directory and run 'ragger rebuild' to re-embed.");
    }

    // Backfill any rows left without embeddings (deferred-embedding writes
    // that didn't get processed before the previous shutdown). The query is
    // a no-op when nothing is NULL; embedder is already loaded above.
    int filled = backend_->backfill_embeddings(*embedder_);
    if (filled > 0) {
        log_info(std::format(lang::MSG_BACKFILLED_EMBEDDINGS, filled));
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

int RaggerMemory::rebuild_bm25() {
    return backend_->rebuild_bm25();
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

} // namespace ragger
