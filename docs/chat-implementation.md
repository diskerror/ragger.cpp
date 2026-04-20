# Chat Verb Implementation — C++ Port Complete

## Summary

Successfully ported the multi-endpoint inference client and chat verb from Python to C++ in the Ragger project.

## What Was Built

### 1. Inference Client (`include/ragger/inference.h` + `src/inference.cpp`)

**Endpoint struct:**
- `name`, `api_url`, `api_key`, `models` (comma-separated glob patterns)
- `bool matches(const std::string& model)` — fnmatch-style glob matching against patterns
- `headers()` — returns HTTP header map; Anthropic uses `x-api-key` + `anthropic-version`, others use `Authorization: Bearer`

**InferenceClient class:**
- `std::vector<Endpoint> _endpoints` — list of configured endpoints
- `std::string model` — default model name
- `int max_tokens` — default token limit (4096)
- `static InferenceClient from_config(const Config& cfg)` — factory method that builds from config
- `Endpoint& resolve_endpoint(const std::string& model)` — finds matching endpoint via glob patterns
- `std::string chat(...)` — blocking chat, returns full response text
- `void chat_stream(...)` — streaming chat with callback for each token

**HTTP Implementation:**
- Uses **libcurl** for HTTP requests (both blocking and streaming)
- SSE (Server-Sent Events) parsing for streaming responses
- Extracts `choices[0].delta.content` from streaming chunks
- Proper error handling for HTTP failures and malformed JSON

### 2. Config Updates (`include/ragger/config.h` + `src/config.cpp`)

**New Config Fields:**
```cpp
struct InferenceEndpointConfig {
    std::string name;
    std::string api_url;
    std::string api_key;
    std::string models = "*";
};

std::string inference_model = "claude-sonnet-4-5";
std::string inference_default = "";
std::string inference_api_url = "";
std::string inference_api_key = "";
int inference_max_tokens = 4096;
std::vector<InferenceEndpointConfig> inference_endpoints;
```

**INI Parsing:**
- `[inference]` section: api_url, api_key, model, max_tokens, default
- `[inference.*]` sections (e.g., `[inference.local]`, `[inference.anthropic]`): api_url, api_key, models
- Section detection with `inference.` prefix to populate endpoints vector
- Default endpoint routing: named default moves to end of list (fallback position)

**Weight Normalization Update:**
- Changed `bm25_weight` default: 0.3 → **3.0**
- Changed `vector_weight` default: 0.7 → **7.0**
- Runtime uses A/(A+B) pattern, so absolute values don't need to sum to 1.0

### 3. Chat Verb (`src/main.cpp`)

**load_workspace_files():**
- SOUL.md, USER.md, AGENTS.md, TOOLS.md: loaded from `~/.ragger/`
- Returns combined text for system prompt injection

**do_chat():**
1. Builds `InferenceClient` from config
2. Loads `RaggerMemory` instance for context search
3. Loads workspace files into initial system message
4. REPL loop:
   - Reads user input from stdin
   - Handles `/quit` and EOF for exit
   - Searches memory (top 3 results, min_score 0.3)
   - Appends memory context to system message
   - Sends to inference API with streaming
   - Prints tokens as they arrive
   - Maintains conversation history
5. Prints model name on startup

**Command Registration:**
- Added `chat` command to main() dispatcher
- Integrated with existing verb-style CLI

### 4. Build System (`CMakeLists.txt`)

**Changes:**
- Added `find_package(CURL REQUIRED)`
- Added `src/inference.cpp` to `ragger_core` sources
- Linked `CURL::libcurl` to `ragger_core`

### 5. Default Config Template

Updated embedded default config in `config.cpp` to include:
- `[inference]` section with examples
- Multi-endpoint setup examples commented out
- Model routing documentation

## Testing

✅ **Build:** Successful compilation with no errors  
✅ **Chat verb exists:** `./ragger chat` launches REPL  
✅ **Config loading:** Reads inference settings from `~/.ragger/settings.ini`  
✅ **Endpoint routing:** Matches model names against glob patterns  
✅ **HTTP client:** libcurl connects to LM Studio on `localhost:1234`  
✅ **Streaming:** SSE parsing extracts tokens and prints them in real-time  
✅ **Memory context:** Searches memory and injects top 3 results into system prompt  
✅ **Conversation history:** Maintains multi-turn dialogue  
✅ **Exit handling:** `/quit` and Ctrl+D both exit cleanly

## Example Session

```bash
$ cd ~/CLionProjects/Ragger/build
$ ./ragger chat
Config loaded from /Volumes/WDBlack2/.ragger/settings.ini
Ragger Chat (model: qwen/qwen2.5-coder-14b)
Type '/quit' or Ctrl+D to exit

You: What is 2+2?
Assistant: 2 + 2 equals 4.

You: /quit
Goodbye!
```

## Config Format

**Single endpoint:**
```ini
[inference]
model = qwen/qwen2.5-coder-14b
api_url = http://localhost:1234/v1
api_key = lmstudio-local
max_tokens = 4096
```

**Multiple endpoints:**
```ini
[inference]
model = claude-sonnet-4-5
default = anthropic

[inference.local]
api_url = http://localhost:1234/v1
api_key = lmstudio-local
models = qwen/*, llama/*, mistral/*

[inference.anthropic]
api_url = https://api.anthropic.com/v1
api_key = sk-ant-...
models = claude-*
```

## Key Constraints Met

✅ **C++17** — Matches existing project standard  
✅ **nlohmann/json** — Uses vendored header at `vendor/nlohmann_json.hpp`  
✅ **libcurl** — Found and linked successfully  
✅ **No Crow for client** — Server-only library not used  
✅ **Same config file** — Uses existing `~/.ragger/settings.ini`  
✅ **Same behavior** — Matches Python implementation exactly  
✅ **No service disruption** — Did not modify running server  
✅ **RaggerMemory instance** — Chat verb loads its own instance like other verbs

## Files Modified

- `include/ragger/config.h` — Added inference config struct and fields
- `src/config.cpp` — Added INI parsing for `[inference]` and `[inference.*]` sections
- `src/main.cpp` — Added `load_workspace_files()`, `do_chat()`, and chat command handler
- `CMakeLists.txt` — Added libcurl dependency and `src/inference.cpp` to build

## Files Created

- `include/ragger/inference.h` — Inference client header
- `src/inference.cpp` — Inference client implementation with libcurl HTTP + SSE streaming

## Binary Location

Built binary: `~/CLionProjects/Ragger/build/ragger` (12 MB)

Ready for testing and deployment.
