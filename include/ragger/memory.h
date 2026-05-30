/**
 * RaggerMemory — high-level facade
 *
 * Mirrors ragger_memory/memory.py from the Python version.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "storage_backend.h"

namespace ragger {

using json = nlohmann::json;

class RaggerMemory {
public:
    /// Construct with optional override for DB path and model dir.
    explicit RaggerMemory(const std::string& db_path = "",
                          const std::string& model_dir = "",
                          const std::string& user_db_path = "");
    ~RaggerMemory();

    /// Store a memory. Stores to the configured DB.
    /// `defer_embedding`: write embedding=NULL; caller (or startup) backfills.
    std::string    store(const std::string& text, json metadata = {},
                         bool common = false,
                         bool defer_embedding = false);

    /// Store a Level 5 RAG document chunk. Pass the same `chunk.imported_at`
    /// for every chunk of one import so they share a single timestamp
    /// (issue #48). Returns the new document id.
    int store_document(const DocumentChunk& chunk, bool defer_embedding = false);

    /// Capture a raw L1 conversation turn into the `turns` table. Empty
    /// assistant_text → partial row (finalize later). Returns turn_id.
    int store_turn(const std::string& user_text,
                   const std::string& assistant_text = "",
                   const std::string& model_name = "",
                   bool defer_embedding = false);
    /// Finalize a partial turn (set assistant_text + embed + model).
    bool finalize_turn(int turn_id,
                       const std::string& assistant_text,
                       const std::string& model_name = "");

    /// Write back a document's embedding (import path).
    bool update_document_embedding(int document_id, const std::vector<float>& emb);

    /// Replace text + metadata of an existing memory (re-embeds unless
    /// `defer_embedding`, rebuilds BM25 tokens, preserves id and original
    /// timestamp). Returns false if the row is missing or has the keep tag.
    bool update_text(int memory_id, const std::string& text, json metadata = {},
                     bool defer_embedding = false);

    /// Search. Returns results from the single configured DB.
    SearchResponse search(const std::string& query,
                          int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});
    /// Total count in the single DB.
    int  count() const;

    /// Load all memories (for export). Optionally filter by collection.
    std::vector<SearchResult> load_all(const std::string& collection = "");

    /// Rebuild BM25 index. Returns doc count.
    int rebuild_bm25();

    /// Rebuild embeddings for all documents. Returns doc count.
    int rebuild_embeddings();

    /// Embed only rows whose embedding column is NULL. Returns count.
    int backfill_embeddings();

    /// Get distinct collection names.
    std::vector<std::string> collections() const;

    /// Delete a memory by ID.
    bool delete_memory(int memory_id);

    /// Delete multiple memories by ID.
    int delete_batch(const std::vector<int>& memory_ids);

    /// Search by metadata field matching.
    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit = 0,
                                                 const std::string& after = "",
                                                 const std::string& before = "");

    /// Access primary backend (for storage operations).
    StorageBackend* backend() { return backend_.get(); }

    void close();

private:
    std::unique_ptr<Embedder>      embedder_;
    std::unique_ptr<StorageBackend> backend_;      // single user DB
};

} // namespace ragger
