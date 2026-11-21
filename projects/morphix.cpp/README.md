# Morphix C++

Native C++ implementation of the Morphix document utility.

## Build

```bash
cmake -S projects/morphix.cpp -B build/morphix.cpp
cmake --build build/morphix.cpp -j
PORT=5034 ./build/morphix.cpp/morphix_cpp
```

The server listens on `0.0.0.0:$PORT` and serves the existing Morphix UI.

Native dependencies: libmicrohttpd, JSONCpp, libcurl, Poppler CPP, libzip,
pugixml, and qpdf.

## Native capabilities

- Native PDF text extraction and text-oriented PDF output
- Native DOCX and EPUB package reading/writing via libzip and pugixml
- Native TXT, Markdown, HTML extraction and conversion
- Native heading-based splitting for text documents
- Native PDF page-range splitting via qpdf
- Native ZIP download assembly
- Claude and Gemini requests via libcurl with provider-specific headers
- Async and synchronous AI split jobs

AZW3 requires conversion to EPUB in the original Python implementation. The
native rewrite does not claim full AZW3/MOBI8 unpacking support.

Ubuntu/proot-distro:

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libmicrohttpd-dev \
  libjsoncpp-dev libcurl4-openssl-dev libpoppler-cpp-dev libzip-dev \
  libpugixml-dev qpdf
```

macOS with Homebrew:

```bash
brew install cmake pkg-config libmicrohttpd jsoncpp curl poppler libzip pugixml qpdf
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:$PKG_CONFIG_PATH"
```
