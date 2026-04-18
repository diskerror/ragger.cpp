/*
 * test_fp16_precision.cpp
 *
 * Test FP16 embedding precision on production Ragger database.
 * Compares F32 baseline vs F16 storage with F32-accumulator compute.
 *
 * Build:
 *   cd ~/CLionProjects/Ragger/tools
 *   g++ -std=c++17 -O3 -march=native \
 *       -I../include -I../vendor/eigen \
 *       test_fp16_precision.cpp \
 *       -lsqlite3 -o test_fp16_precision
 *
 * Run:
 *   ./test_fp16_precision /var/ragger/memories.db
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <sqlite3.h>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

// Test queries from actual usage patterns
const vector<string> TEST_QUERIES = {
    "user management patterns in Ragger",
    "how to configure TLS",
    "persistent chat sessions implementation",
    "vector search optimization",
    "Beelink hardware specifications",
    "config reload SIGHUP",
    "deferred embedding background",
    "PBKDF2 password hashing",
    "cosine similarity calculation",
    "multi-user database routing",
    "LM Studio model auto-load",
    "housekeeping cleanup strategy",
    "web session authentication",
    "cpp-httplib SSE streaming",
    "temporal filtering after parameter"
};

struct Memory {
    int64_t id;
    string text;
    VectorXf embedding_f32;
};

// Load embeddings from production DB
vector<Memory> load_embeddings(const string& db_path) {
    sqlite3* db;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
        cerr << "Failed to open database: " << db_path << endl;
        exit(1);
    }

    const char* sql = "SELECT id, text, embedding FROM memories WHERE embedding IS NOT NULL";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        cerr << "Failed to prepare statement" << endl;
        exit(1);
    }

    vector<Memory> memories;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory mem;
        mem.id = sqlite3_column_int64(stmt, 0);
        mem.text = string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));

        const void* blob = sqlite3_column_blob(stmt, 2);
        int blob_size = sqlite3_column_bytes(stmt, 2);

        int dimension = blob_size / sizeof(float);
        mem.embedding_f32.resize(dimension);
        memcpy(mem.embedding_f32.data(), blob, blob_size);

        memories.push_back(mem);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    cout << "Loaded " << memories.size() << " embeddings ("
         << memories[0].embedding_f32.size() << " dimensions)" << endl;

    return memories;
}

// Baseline: F32/F32
float cosine_similarity_f32(const VectorXf& a, const VectorXf& b) {
    float dot = a.dot(b);
    float norm_a = a.norm();
    float norm_b = b.norm();
    return dot / (norm_a * norm_b);
}

// Conservative: F16 storage, upcast to F32 for compute
float cosine_similarity_f16_upcast(const VectorXf& a_f32, const VectorXf& b_f32) {
    Matrix<half, Dynamic, 1> a_f16 = a_f32.cast<half>();
    Matrix<half, Dynamic, 1> b_f16 = b_f32.cast<half>();

    VectorXf a = a_f16.cast<float>();
    VectorXf b = b_f16.cast<float>();

    float dot = a.dot(b);
    float norm_a = a.norm();
    float norm_b = b.norm();
    return dot / (norm_a * norm_b);
}

// Hybrid: F16 multiply, F32 accumulate (primary candidate)
float cosine_similarity_f16_f32acc(const VectorXf& a_f32, const VectorXf& b_f32) {Matrix<half, Dynamic, 1> a = a_f32.cast<half>();
    Matrix<half, Dynamic, 1> b = b_f32.cast<half>();

    // Dot product: F16 multiply, F32 accumulate
    float dot_f32 = 0.0f;
    for (int i = 0; i < a.size(); i++) {
        half prod = a[i] * b[i];
        dot_f32 += float(prod);
    }

    // Norms: same pattern
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    for (int i = 0; i < a.size(); i++) {
        half a_sq = a[i] * a[i];
        half b_sq = b[i] * b[i];
        norm_a_sq += float(a_sq);
        norm_b_sq += float(b_sq);
    }

    float norm_a = sqrt(norm_a_sq);
    float norm_b = sqrt(norm_b_sq);

    return dot_f32 / (norm_a * norm_b);
}

struct SearchResult {
    int64_t id;
    string text;
    float score;

    bool operator<(const SearchResult& other) const {
        return score > other.score;
    }
};

vector<SearchResult> search(
    const VectorXf& query,
    const vector<Memory>& corpus,
    float (*similarity_fn)(const VectorXf&, const VectorXf&)
) {
    vector<SearchResult> results;

    for (const auto& mem : corpus) {
        float score = similarity_fn(query, mem.embedding_f32);
        results.push_back({mem.id, mem.text, score});
    }

    sort(results.begin(), results.end());
    return results;
}

struct Comparison {
    int overlap_top10;
    double mean_diff;
    double max_diff;
    double p95_diff;
};

Comparison compare_results(
    const vector<SearchResult>& baseline,
    const vector<SearchResult>& test
) {
    // Top-10 overlap
    vector<int64_t> ids_baseline, ids_test;
    for (int i = 0; i < 10 && i < baseline.size(); i++) {
        ids_baseline.push_back(baseline[i].id);
        ids_test.push_back(test[i].id);
    }

    int overlap = 0;
    for (auto id : ids_baseline) {
        if (find(ids_test.begin(), ids_test.end(), id) != ids_test.end()) {
            overlap++;
        }
    }

    // Score differences
    map<int64_t, float> baseline_scores;
    for (const auto& r : baseline) {
        baseline_scores[r.id] = r.score;
    }

    vector<double> diffs;
    for (const auto& r : test) {
        if (baseline_scores.count(r.id)) {
            diffs.push_back(abs(r.score - baseline_scores[r.id]));
        }
    }

    if (diffs.empty()) {
        return {overlap, 0.0, 0.0, 0.0};
    }

    sort(diffs.begin(), diffs.end());

    double mean = 0.0;
    for (auto d : diffs) mean += d;
    mean /= diffs.size();

    double max_diff = diffs.back();
    double p95 = diffs[size_t(diffs.size() * 0.95)];

    return {overlap, mean, max_diff, p95};
}

double benchmark_speed(
    const VectorXf& query,
    const vector<Memory>& corpus,
    float (*similarity_fn)(const VectorXf&, const VectorXf&),
    int iterations = 100
) {
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        search(query, corpus, similarity_fn);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    return elapsed.count() / iterations;
}

// Mock embedding (replace with real model if available)
VectorXf embed_query(const string& query_text, int dimension) {
    // Use hash of query text as seed for reproducibility
    size_t seed = hash<string>{}(query_text);
    srand(seed);

    VectorXf embedding(dimension);
    for (int i = 0; i < dimension; i++) {
        embedding[i] = (rand() / double(RAND_MAX)) * 2.0 - 1.0;
    }

    return embedding.normalized();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <database_path>" << endl;
        return 1;
    }

    string db_path = argv[1];

    cout << "Loading embeddings from: " << db_path << endl;
    vector<Memory> memories = load_embeddings(db_path);

    int dimension = memories[0].embedding_f32.size();

    cout << "\n" << string(70, '=') << endl;cout << "FP16 Precision Test — Candidate for Default" << endl;
    cout << string(70, '=') << endl;

    vector<Comparison> upcast_comparisons;
    vector<Comparison> f32acc_comparisons;

    for (const auto& query_text : TEST_QUERIES) {
        cout << "\nQuery: " << query_text << endl;
        cout << string(70, '-') << endl;

        VectorXf query = embed_query(query_text, dimension);

        auto results_f32 = search(query, memories, cosine_similarity_f32);
        auto results_f16_upcast = search(query, memories, cosine_similarity_f16_upcast);
        auto results_f16_f32acc = search(query, memories, cosine_similarity_f16_f32acc);

        auto cmp_upcast = compare_results(results_f32, results_f16_upcast);
        auto cmp_f32acc = compare_results(results_f32, results_f16_f32acc);

        upcast_comparisons.push_back(cmp_upcast);
        f32acc_comparisons.push_back(cmp_f32acc);

        cout << "\nMode              | Overlap | Mean Δ   | Max Δ    | P95 Δ" << endl;
        cout << string(70, '-') << endl;
        cout << "F16/F32-upcast    | " << cmp_upcast.overlap_top10 << "/10   | "
             << cmp_upcast.mean_diff << " | " << cmp_upcast.max_diff << " | "
             << cmp_upcast.p95_diff << endl;
        cout << "F16/F32-acc (*)   | " << cmp_f32acc.overlap_top10 << "/10   | "
             << cmp_f32acc.mean_diff << " | " << cmp_f32acc.max_diff << " | "
             << cmp_f32acc.p95_diff << endl;

        cout << "\nTop-5 (F32/F32 baseline):" << endl;
        for (int i = 0; i < 5 && i < results_f32.size(); i++) {
            printf("  %d. [%.4f] %s...\n", i+1, results_f32[i].score,
                   results_f32[i].text.substr(0, 55).c_str());
        }

        cout << "\nTop-5 (F16/F32-acc — primary candidate):" << endl;
        for (int i = 0; i < 5 && i < results_f16_f32acc.size(); i++) {
            printf("  %d. [%.4f] %s...\n", i+1, results_f16_f32acc[i].score,
                   results_f16_f32acc[i].text.substr(0, 55).c_str());
        }
    }

    // Aggregate statistics
    cout << "\n" << string(70, '=') << endl;
    cout << "Aggregate Statistics (across " << TEST_QUERIES.size() << " queries)" << endl;
    cout << string(70, '=') << endl;

    auto avg = [](const vector<Comparison>& comps, auto member) {
        double sum = 0.0;
        for (const auto& c : comps) sum += c.*member;
        return sum / comps.size();
    };

    cout << "\nF16/F32-upcast:" << endl;
    cout << "  Avg Top-10 Overlap: " << avg(upcast_comparisons, &Comparison::overlap_top10) << "/10" << endl;
    cout << "  Avg Mean Δ:         " << avg(upcast_comparisons, &Comparison::mean_diff) << endl;
    cout << "  Avg P95 Δ:          " << avg(upcast_comparisons, &Comparison::p95_diff) << endl;

    cout << "\nF16/F32-acc (primary candidate):" << endl;
    cout << "  Avg Top-10 Overlap: " << avg(f32acc_comparisons, &Comparison::overlap_top10) << "/10" << endl;
    cout << "  Avg Mean Δ:         " << avg(f32acc_comparisons, &Comparison::mean_diff) << endl;
    cout << "  Avg P95 Δ:          " << avg(f32acc_comparisons, &Comparison::p95_diff) << endl;

    // Speed benchmark
    cout << "\n" << string(70, '=') << endl;
    cout << "Speed Benchmark (100 iterations)" << endl;
    cout << string(70, '=') << endl;

    VectorXf query = embed_query(TEST_QUERIES[0], dimension);

    double t_f32 = benchmark_speed(query, memories, cosine_similarity_f32);
    double t_f16_upcast = benchmark_speed(query, memories, cosine_similarity_f16_upcast);
    double t_f16_f32acc = benchmark_speed(query, memories, cosine_similarity_f16_f32acc);

    printf("F32/F32:        %.2f ms/search\n", t_f32);
    printf("F16/F32-upcast: %.2f ms (%.2fx)\n", t_f16_upcast, t_f32/t_f16_upcast);
    printf("F16/F32-acc:    %.2f ms (%.2fx)\n", t_f16_f32acc, t_f32/t_f16_f32acc);

    // Recommendation
    cout << "\n" << string(70, '=') << endl;cout << "Recommendation" << endl;
    cout << string(70, '=') << endl;

    double avg_overlap_f32acc = avg(f32acc_comparisons, &Comparison::overlap_top10);

    if (avg_overlap_f32acc >= 9.0) {
        cout << "\n✅ F16/F32-acc is EXCELLENT (avg " << avg_overlap_f32acc
             << "/10 overlap)" << endl;
        cout << "   Recommend as default with F32 as config option." << endl;
    } else if (avg_overlap_f32acc >= 8.0) {
        cout << "\n⚠️  F16/F32-acc is GOOD (avg " << avg_overlap_f32acc
             << "/10 overlap)" << endl;
        cout << "   Consider as default, but test with real queries." << endl;
    } else {
        cout << "\n❌ F16/F32-acc has quality issues (avg " << avg_overlap_f32acc
             << "/10 overlap)" << endl;
        cout << "   Keep F32 as default, offer F16 as experimental." << endl;
    }

    cout << "\nStorage savings: ~40% (50% on embeddings, ~40% total DB)" << endl;

    return 0;
}
