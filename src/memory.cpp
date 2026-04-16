/**
 * RaggerMemory — high-level facade implementation
 */

#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "ragger/sqlite_user_manager.h"
#include "ragger/embedder.h"
#include "ragger/config.h"
#include "ragger/logs.h"

#include <filesystem>
#include <pwd.h>

namespace ragger {
namespace fs = std::filesystem;

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

    // Create user manager on the same DB as the primary backend
    user_manager_ = std::make_unique<SqliteUserManager>(backend_->db_path());

    // Multi-user: create user backend
    if (!user_db_path.empty()) {
        user_backend_ = std::make_unique<SqliteBackend>(*embedder_, user_db_path);
    }
}

// Private constructor for for_user() — shares embedder and common backend
RaggerMemory::RaggerMemory(Embedder* shared_embedder, StorageBackend* shared_common,
                           const std::string& user_db_path)
    : shared_embedder_(shared_embedder)
{
    // backend_ points to shared common (non-owning would be cleaner but
    // we wrap it in a unique_ptr with a no-op deleter isn't worth the complexity).
    // Instead, we create a new SqliteBackend for common that shares the same DB.
    // Actually — just store a raw pointer and special-case close().
    // Simplest: re-open common DB read-only... No, that breaks store-to-common.
    //
    // Real approach: backend_ wraps the same DB path with shared embedder.
    // The common DB supports concurrent readers via WAL mode.
    backend_ = std::make_unique<SqliteBackend>(*shared_embedder, shared_common->db_path());
    user_backend_ = std::make_unique<SqliteBackend>(*shared_embedder, user_db_path);
    user_manager_ = std::make_unique<SqliteUserManager>(backend_->db_path());
}

std::string RaggerMemory::resolve_user_home(const std::string& username) {
    // Try pwd first (works on macOS and Linux)
    struct passwd* pw = getpwnam(username.c_str());
    if (pw && pw->pw_dir) {
        std::string home = pw->pw_dir;
        if (fs::is_directory(home)) return home;
    }

    // Fallback: common home base directories
#ifdef __APPLE__
    std::string candidate = "/Users/" + username;
#else
    std::string candidate = "/home/" + username;
#endif
    if (fs::is_directory(candidate)) return candidate;

    return "";
}

std::unique_ptr<RaggerMemory> RaggerMemory::for_user(const std::string& username) {
    std::string home = resolve_user_home(username);
    if (home.empty()) {
        log_info("[WARN] Cannot resolve home for user '" + username + "', using common DB only");
        return nullptr;
    }

    std::string user_db = home + "/.ragger/memories.db";
    if (!fs::exists(user_db)) {
        log_info("No user DB for '" + username + "' at " + user_db + ", using common DB only");
        return nullptr;
    }

    Embedder* emb = shared_embedder_ ? shared_embedder_ : embedder_.get();
    auto inst = std::unique_ptr<RaggerMemory>(
        new RaggerMemory(emb, backend_.get(), user_db));
    log_info("Opened user DB for '" + username + "': " + user_db);
    return inst;
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

std::vector<SearchResult> RaggerMemory::search_by_metadata(const json& metadata_filter, int limit,
                                                           const std::string& after,
                                                           const std::string& before) {
    if (!user_backend_) {
        return backend_->search_by_metadata(metadata_filter, limit, after, before);
    }

    // Multi-DB: query both, merge results
    auto common_results = backend_->search_by_metadata(metadata_filter, limit, after, before);
    auto user_results = user_backend_->search_by_metadata(metadata_filter, limit, after, before);

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
