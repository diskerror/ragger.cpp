/**
 * RaggerMemory — high-level facade implementation
 */

#include "ragger/memory.h"
#include "ragger/embedder.h"
#include "ragger/config.h"

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

    // Create primary backend (common DB in multi-user, only DB in single-user)
    backend_ = std::make_unique<SqliteBackend>(*embedder_, db_path);

    // Multi-user: create user backend
    if (!user_db_path.empty()) {
        user_backend_ = std::make_unique<SqliteBackend>(*embedder_, user_db_path);
    }
}

RaggerMemory::~RaggerMemory() {
    close();
}

std::string RaggerMemory::store(const std::string& text, json metadata,
                                 bool common) {
    // Auto-set "keep" tag when storing to common DB
    if (common) {
        metadata["keep"] = true;
    }
    
    if (user_backend_ && !common) {
        return user_backend_->store(text, std::move(metadata));
    }
    return backend_->store(text, std::move(metadata));
}

SearchResponse RaggerMemory::search(const std::string& query,
                                    int limit,
                                    float min_score,
                                    std::vector<std::string> collections) {
    if (!user_backend_) {
        return backend_->search(query, limit, min_score, std::move(collections));
    }

    // Multi-DB: query both, merge by score
    auto common_result = backend_->search(query, limit, min_score, collections);
    auto user_result = user_backend_->search(query, limit, min_score, std::move(collections));

    // Merge results
    auto& all = common_result.results;
    all.insert(all.end(), user_result.results.begin(), user_result.results.end());
    std::sort(all.begin(), all.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });
    if ((int)all.size() > limit) all.resize(limit);

    // Merge timing
    auto& ct = common_result.timing;
    auto& ut = user_result.timing;
    ct["search_ms"] = ct.value("search_ms", 0) + ut.value("search_ms", 0);
    ct["total_ms"] = ct.value("total_ms", 0) + ut.value("total_ms", 0);
    ct["corpus_size"] = ct.value("corpus_size", 0) + ut.value("corpus_size", 0);

    return common_result;
}

int RaggerMemory::count() const {
    int total = backend_->count();
    if (user_backend_) total += user_backend_->count();
    return total;
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

std::vector<std::string> RaggerMemory::collections() const {
    return backend_->collections();
}

bool RaggerMemory::delete_memory(int memory_id) {
    // In multi-DB mode: try user DB first, then common
    if (user_backend_) {
        if (user_backend_->delete_memory(memory_id)) {
            return true;
        }
    }
    return backend_->delete_memory(memory_id);
}

int RaggerMemory::delete_batch(const std::vector<int>& memory_ids) {
    int total = 0;
    // In multi-DB mode: try both DBs
    if (user_backend_) {
        total += user_backend_->delete_batch(memory_ids);
    }
    total += backend_->delete_batch(memory_ids);
    return total;
}

std::vector<SearchResult> RaggerMemory::search_by_metadata(const json& metadata_filter, int limit) {
    if (!user_backend_) {
        return backend_->search_by_metadata(metadata_filter, limit);
    }

    // Multi-DB: query both, merge results
    auto common_results = backend_->search_by_metadata(metadata_filter, limit);
    auto user_results = user_backend_->search_by_metadata(metadata_filter, limit);

    // Merge results
    common_results.insert(common_results.end(), user_results.begin(), user_results.end());
    
    // Sort by ID for consistency
    std::sort(common_results.begin(), common_results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.id < b.id;
              });

    // Apply limit if specified
    if (limit > 0 && (int)common_results.size() > limit) {
        common_results.resize(limit);
    }

    return common_results;
}

void RaggerMemory::close() {
    if (backend_) backend_->close();
    if (user_backend_) user_backend_->close();
}

} // namespace ragger
