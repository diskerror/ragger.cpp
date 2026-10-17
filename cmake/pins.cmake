# pins.cmake — exact versions of fetched dependencies (see Deps.cmake).
# Update by running a -DDEPS_LATEST=ON configure, then scripts/pin-deps.sh.
# Don't hand-edit the *_SHA lines unless you mean to.

set(TOKENIZERS_CPP_REPO https://github.com/mlc-ai/tokenizers-cpp.git)
set(TOKENIZERS_CPP_SHA  c586c52f93f7b060753bd2388eb96a105cb7374d)

set(C_LIB_REPO https://github.com/diskerror/c_lib.git)
set(C_LIB_SHA  9814f0600d6985d27dc30a54925dfd2fc7afabdb)

# ONNX Runtime ships as release binaries, not a git repo: pin the version in the
# top-level CMakeLists (ORT_VERSION) and its archive checksums here. Bump by
# hand — a new major version can change the ABI. (GitHub's published asset
# digests; osx-x86_64 has no 1.24.4 build, so it downloads unverified.)
set(ORT_SHA256_onnxruntime-linux-x64-1.24.4     3a211fbea252c1e66290658f1b735b772056149f28321e71c308942cdb54b747)
set(ORT_SHA256_onnxruntime-linux-aarch64-1.24.4 866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e)
set(ORT_SHA256_onnxruntime-osx-arm64-1.24.4     93787795f47e1eee369182e43ed51b9e5da0878ab0346aecf4258979b8bba989)
