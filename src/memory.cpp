/**
 * RaggerMemory — high-level facade implementation
 */

#include "ragger/memory.h"
#include "ragger/embedder.h"
#include "ragger/config.h"

namespace ragger {

RaggerMemory::RaggerMemory(const std::string& db_path,
                           const std::string& model_dir)
{
    // Resolve model directory from config or override
    std::string resolved_model_dir = model_dir.empty()
        ? config().resolved_model_dir()
        : expand_path(model_dir);

    // Create embedder
    embedder_ = std::make_unique<Embedder>(resolved_model_dir);

    // Create SQLite backend with embedder reference
    backend_ = std::make_unique<SqliteBackend>(*embedder_, db_path);
}

RaggerMemory::~RaggerMemory() {
    close();
}

std::string RaggerMemory::store(const std::string& text, json metadata) {
    return backend_->store(text, std::move(metadata));
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

std::vector<std::string> RaggerMemory::collections() const {
    return backend_->collections();
}

void RaggerMemory::close() {
    if (backend_) {
        backend_->close();
    }
}

} // namespace ragger
