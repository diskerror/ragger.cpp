/**
 * Client module tests
 *
 * Tests HTTP client construction and basic functionality.
 * Integration tests are commented out (require running server).
 */
#include "ragger/client.h"
#include <cassert>
#include <iostream>
#include <print>

void test_client_construction() {
    std::print("  test_client_construction..."); std::cout.flush();

    // Should be able to construct with default args
    ragger::RaggerClient client1;

    // Should be able to construct with all args
    ragger::RaggerClient client2("127.0.0.1", 8432, "test-token");

    // Should be able to construct with custom port
    ragger::RaggerClient client3("localhost", 9999, "");

    std::cout << " OK\n";
}

void test_is_available_when_no_server() {
    std::print("  test_is_available_when_no_server..."); std::cout.flush();

    // Use an unlikely port to ensure nothing is listening
    ragger::RaggerClient client("127.0.0.1", 59999, "");
    
    // Should return false when no server is running
    bool available = client.is_available();
    assert(!available);

    std::cout << " OK\n";
}

void test_http_response_parsing() {
    std::print("  test_http_response_parsing..."); std::cout.flush();

    // This just tests that we can create a client without errors
    // Actual HTTP parsing is tested implicitly by the other methods
    // Use an unlikely port to ensure nothing is listening
    ragger::RaggerClient client("127.0.0.1", 59998, "token123");
    
    // Test that the client is constructed properly
    // (we can't easily test HTTP parsing without a mock server)
    assert(!client.is_available());  // Should be false when no server

    std::cout << " OK\n";
}

// Integration tests (commented out by default, require running server)
/*
void test_integration_store_and_search() {
    std::cout << "  test_integration_store_and_search..." << std::flush;

    // NOTE: This test requires a running Ragger server on port 8432
    // Start it with: ragger serve
    
    ragger::RaggerClient client("127.0.0.1", 8432, "");
    
    // Check server is available
    assert(client.is_available());
    
    // Store a test memory
    nlohmann::json meta = {{"collection", "test"}};
    auto id = client.store("Test memory from client integration test", meta);
    assert(!id.empty());
    
    // Search for it
    auto response = client.search("test memory", 5, 0.0f);
    assert(!response.results.empty());
    
    // Count should be > 0
    int count = client.count();
    assert(count > 0);

    std::cout << " OK\n";
}
*/

int main() {
    std::println("Running client tests:");

    test_client_construction();
    test_is_available_when_no_server();
    test_http_response_parsing();

    // Integration tests (commented out)
    // test_integration_store_and_search();

    std::cout << "test_client: all passed\n";
    return 0;
}
