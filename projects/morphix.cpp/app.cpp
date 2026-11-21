#include <microhttpd.h>
#include <json/json.h>
#include <curl/curl.h>
#include <poppler-document.h>
#include <poppler-page.h>
#include <pugixml.hpp>
#include <zip.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <ctime>
#include <unordered_map>
#include <set>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

struct Record {
    std::string id;
    std::string filename;
    std::string ext;
    fs::path path;
};

struct ConnectionInfo {
    MHD_PostProcessor* processor = nullptr;
    std::string upload_filename;
    std::string upload_data;
    std::string json_body;
    std::vector<std::pair<std::string, std::string>> uploads;
    std::string boundary;
    bool first_callback = true;
};

static fs::path storage_dir;
static std::unordered_map<std::string, Record> files;
static std::unordered_map<std::string, Record> outputs;
struct Job {
    std::string status = "running";
    std::string phase = "starting";
    std::string error;
    int pct = 0;
    Json::Value chapters;
    bool truncated = false;
};
static std::unordered_map<std::string, Job> jobs;
static std::mutex jobs_mutex;

static std::string random_id() {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 32; ++i) out += hex[(rng() >> (i % 8)) & 15];
    return out;
}

static std::string json_escape(const std::string& value) {
    Json::Value json(value);
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    std::string encoded = Json::writeString(builder, json);
    if (encoded.size() >= 2 && encoded.front() == '"' && encoded.back() == '"')
        encoded = encoded.substr(1, encoded.size() - 2);
    return encoded;
}

static std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static bool write_file(const fs::path& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return out.good();
}

static std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return value;
}

static std::string trim(const std::string& value) {
    const char* ws = " \t\r\n";
    size_t begin = value.find_first_not_of(ws);
    if (begin == std::string::npos) return {};
    size_t end = value.find_last_not_of(ws);
    return value.substr(begin, end - begin + 1);
}

static std::string safe_stem(const std::string& filename) {
    std::string stem = fs::path(filename).stem().string();
    std::string out;
    for (unsigned char c : stem) {
        if (std::isalnum(c) || c == '-' || c == '_') out += static_cast<char>(c);
        else if (std::isspace(c) && (out.empty() || out.back() != '_')) out += '_';
    }
    return out.empty() ? "document" : out;
}

static std::string extension(const std::string& filename) {
    std::string ext = fs::path(filename).extension().string();
    if (!ext.empty()) ext.erase(ext.begin());
    return lower(ext);
}

static bool allowed_extension(const std::string& ext) {
    static const std::vector<std::string> values =
        {"pdf", "docx", "epub", "azw3", "txt", "md", "markdown", "html", "htm"};
    return std::find(values.begin(), values.end(), ext) != values.end();
}

static std::string zip_text(const fs::path& path, const std::regex& name_filter) {
    int error = 0;
    zip_t* archive = zip_open(path.c_str(), ZIP_RDONLY, &error);
    if (!archive) return {};
    std::string result;
    zip_int64_t count = zip_get_num_entries(archive, 0);
    for (zip_int64_t i = 0; i < count; ++i) {
        zip_stat_t stat;
        zip_stat_init(&stat);
        if (zip_stat_index(archive, i, 0, &stat) != 0) continue;
        if (!std::regex_search(std::string(stat.name), name_filter)) continue;
        zip_file_t* file = zip_fopen_index(archive, i, 0);
        if (!file) continue;
        char buffer[16384];
        zip_int64_t read;
        while ((read = zip_fread(file, buffer, sizeof(buffer))) > 0)
            result.append(buffer, static_cast<size_t>(read));
        zip_fclose(file);
    }
    zip_close(archive);
    return result;
}

static std::string xml_text(pugi::xml_document& document) {
    std::string result;
    for (pugi::xpath_node node : document.select_nodes("//*[self::w:t]")) {
        result += node.node().child_value();
        result += "\n";
    }
    return trim(result);
}

static std::string strip_html(const std::string& html) {
    std::string text = std::regex_replace(html, std::regex("<[^>]*>"), "\n");
    return trim(std::regex_replace(text, std::regex("[ \\t\\r\\n]+"), " "));
}

static std::string pdf_text(const fs::path& path) {
    std::unique_ptr<poppler::document> document(
        poppler::document::load_from_file(path.string()));
    if (!document || document->is_locked()) return {};
    std::string result;
    for (int i = 0; i < document->pages(); ++i) {
        std::unique_ptr<poppler::page> page(document->create_page(i));
        if (!page) continue;
        poppler::byte_array bytes = page->text().to_utf8();
        result.append(bytes.begin(), bytes.end());
        result += "\n";
    }
    return trim(result);
}

static std::string extract_docx(const fs::path& path) {
    std::string xml = zip_text(path, std::regex(R"(^(word/)?document\.xml$)",
                                                 std::regex::icase));
    pugi::xml_document document;
    if (!document.load_string(xml.c_str())) return {};
    std::string result;
    pugi::xpath_node_set paragraphs =
        document.select_nodes("//*[local-name()='p']");
    for (pugi::xpath_node paragraph : paragraphs) {
        std::string line;
        pugi::xpath_node_set text_nodes =
            paragraph.node().select_nodes(".//*[local-name()='t']");
        for (pugi::xpath_node node : text_nodes)
            line += node.node().child_value();
        if (!trim(line).empty()) result += trim(line) + "\n";
    }
    return trim(result);
}

static std::string extract_epub(const fs::path& path) {
    std::string html = zip_text(path, std::regex(R"(\.(xhtml|html|htm)$)",
                                                 std::regex::icase));
    return strip_html(html);
}

static std::string extract_text(const fs::path& path, const std::string& ext) {
    if (ext == "pdf") return pdf_text(path);
    if (ext == "docx") return extract_docx(path);
    if (ext == "epub") return extract_epub(path);
    std::string text = read_file(path);
    if (ext == "html" || ext == "htm") return strip_html(text);
    return trim(text);
}

static std::string render_text(const std::string& text,
                               const std::string& target,
                               const std::string& title) {
    if (target == "txt") return text;
    if (target == "md") return text;
    if (target == "html") {
        std::string escaped;
        for (char c : text) {
            if (c == '&') escaped += "&amp;";
            else if (c == '<') escaped += "&lt;";
            else if (c == '>') escaped += "&gt;";
            else if (c == '"') escaped += "&quot;";
            else escaped += c;
        }
        return "<!doctype html><html><head><meta charset=\"utf-8\"><title>" +
               title + "</title></head><body><pre>" + escaped +
               "</pre></body></html>";
    }
    return {};
}

static std::string pdf_escape(const std::string& value) {
    std::string out;
    for (unsigned char c : value) {
        if (c == '\\' || c == '(' || c == ')') out += '\\';
        if (c < 32 || c > 126) out += '?';
        else out += static_cast<char>(c);
    }
    return out;
}

static bool render_pdf(const std::string& text, const fs::path& path) {
    std::vector<std::string> lines;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) lines.push_back(line);
    if (lines.empty()) lines.emplace_back("");

    std::vector<std::string> pages;
    const size_t lines_per_page = 48;
    for (size_t start = 0; start < lines.size(); start += lines_per_page) {
        size_t end = std::min(lines.size(), start + lines_per_page);
        std::ostringstream stream;
        stream << "BT /F1 11 Tf 50 790 Td 14 TL\n";
        for (size_t i = start; i < end; ++i)
            stream << "(" << pdf_escape(lines[i]) << ") Tj T*\n";
        stream << "ET";
        pages.push_back(stream.str());
    }

    std::string pdf = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n";
    size_t font_object = pages.size() * 2 + 3;
    std::vector<size_t> offsets(font_object + 1, 0);
    offsets[1] = pdf.size();
    pdf += "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n";
    offsets[2] = pdf.size();
    pdf += "2 0 obj << /Type /Pages /Kids [";
    for (size_t i = 0; i < pages.size(); ++i) {
        pdf += std::to_string(3 + i * 2) + " 0 R ";
    }
    pdf += "] /Count " + std::to_string(pages.size()) +
           " >> endobj\n";

    for (size_t i = 0; i < pages.size(); ++i) {
        size_t page_object = 3 + i * 2;
        size_t stream_object = page_object + 1;
        offsets[page_object] = pdf.size();
        pdf += std::to_string(page_object) +
               " 0 obj << /Type /Page /Parent 2 0 R "
               "/MediaBox [0 0 612 842] /Resources "
               "<< /Font << /F1 " + std::to_string(pages.size() * 2 + 3) +
               " 0 R >> >> /Contents " + std::to_string(stream_object) +
               " 0 R >> endobj\n";
        offsets[stream_object] = pdf.size();
        pdf += std::to_string(stream_object) +
               " 0 obj << /Length " + std::to_string(pages[i].size()) +
               " >> stream\n" + pages[i] + "\nendstream endobj\n";
    }

    offsets[font_object] = pdf.size();
    pdf += std::to_string(font_object) +
           " 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n";
    size_t xref_offset = pdf.size();
    pdf += "xref\n0 " + std::to_string(font_object + 1) + "\n";
    pdf += "0000000000 65535 f \n";
    for (size_t i = 1; i <= font_object; ++i) {
        char line[22];
        size_t offset = i < offsets.size() ? offsets[i] : 0;
        std::snprintf(line, sizeof(line), "%010zu 00000 n \n", offset);
        pdf += line;
    }
    pdf += "trailer << /Size " + std::to_string(font_object + 1) +
           " /Root 1 0 R >>\nstartxref\n" +
           std::to_string(xref_offset) + "\n%%EOF\n";
    return write_file(path, pdf);
}

static bool zip_add_text(zip_t* archive, const char* name,
                         const std::string& text) {
    char* owned = static_cast<char*>(std::malloc(std::max<size_t>(text.size(), 1)));
    if (!owned) return false;
    if (!text.empty()) std::memcpy(owned, text.data(), text.size());
    zip_source_t* source = zip_source_buffer(
        archive, owned, text.size(), 1);
    if (!source) {
        std::free(owned);
        return false;
    }
    if (zip_file_add(archive, name, source, ZIP_FL_OVERWRITE) < 0) {
        zip_source_free(source);
        return false;
    }
    return true;
}

static bool render_docx(const std::string& text, const fs::path& path) {
    int error = 0;
    zip_t* archive = zip_open(path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
    if (!archive) return false;
    const char* types =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "</Types>";
    const char* rels =
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/></Relationships>";
    std::ostringstream document;
    document << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>";
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        document << "<w:p><w:r><w:t xml:space=\"preserve\">";
        for (char c : line) {
            if (c == '&') document << "&amp;";
            else if (c == '<') document << "&lt;";
            else if (c == '>') document << "&gt;";
            else document << c;
        }
        document << "</w:t></w:r></w:p>";
    }
    document << "<w:sectPr/></w:body></w:document>";
    bool ok = zip_add_text(archive, "[Content_Types].xml", types) &&
              zip_add_text(archive, "_rels/.rels", rels) &&
              zip_add_text(archive, "word/document.xml", document.str());
    if (zip_close(archive) != 0) {
        zip_discard(archive);
        return false;
    }
    return ok;
}

static bool render_epub(const std::string& text, const fs::path& path,
                        const std::string& title) {
    int error = 0;
    zip_t* archive = zip_open(path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
    if (!archive) return false;
    std::string mimetype = "application/epub+zip";
    bool mimetype_added = zip_add_text(archive, "mimetype", mimetype);
    zip_int64_t mimetype_index = zip_name_locate(archive, "mimetype", 0);
    if (mimetype_added && mimetype_index >= 0)
        zip_set_file_compression(archive, mimetype_index, ZIP_CM_STORE, 0);
    const char* container =
        "<?xml version=\"1.0\"?><container version=\"1.0\" "
        "xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\">"
        "<rootfiles><rootfile full-path=\"OEBPS/content.opf\" "
        "media-type=\"application/oebps-package+xml\"/></rootfiles></container>";
    std::ostringstream content;
    content << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<package xmlns=\"http://www.idpf.org/2007/opf\" unique-identifier=\"bookid\" version=\"2.0\">"
            "<metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>"
            << title << "</dc:title><dc:language>en</dc:language></metadata>"
            "<manifest><item id=\"chapter\" href=\"chapter.xhtml\" "
            "media-type=\"application/xhtml+xml\"/><item id=\"ncx\" href=\"toc.ncx\" "
            "media-type=\"application/x-dtbncx+xml\"/></manifest>"
            "<spine toc=\"ncx\"><itemref idref=\"chapter\"/></spine></package>";
    std::string escaped;
    for (char c : text) {
        if (c == '&') escaped += "&amp;";
        else if (c == '<') escaped += "&lt;";
        else if (c == '>') escaped += "&gt;";
        else escaped += c;
    }
    std::string chapter =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><html xmlns=\"http://www.w3.org/1999/xhtml\">"
        "<head><title>" + title + "</title></head><body><pre>" +
        escaped + "</pre></body></html>";
    std::string ncx =
        "<?xml version=\"1.0\"?><ncx xmlns=\"http://www.daisy.org/z3986/2005/ncx/\" version=\"2005-1\">"
        "<head/><docTitle><text>" + title + "</text></docTitle>"
        "<navMap><navPoint id=\"chapter\" playOrder=\"1\"><navLabel><text>Chapter</text></navLabel>"
        "<content src=\"chapter.xhtml\"/></navPoint></navMap></ncx>";
    bool ok = mimetype_added &&
        zip_add_text(archive, "META-INF/container.xml", container) &&
        zip_add_text(archive, "OEBPS/content.opf", content.str()) &&
        zip_add_text(archive, "OEBPS/chapter.xhtml", chapter) &&
        zip_add_text(archive, "OEBPS/toc.ncx", ncx);
    if (zip_close(archive) != 0) {
        zip_discard(archive);
        return false;
    }
    return ok;
}

static bool render_document(const std::string& text, const std::string& target,
                            const fs::path& path, const std::string& title) {
    if (target == "txt" || target == "md" || target == "html")
        return write_file(path, render_text(text, target, title));
    if (target == "pdf") return render_pdf(text, path);
    if (target == "docx") return render_docx(text, path);
    if (target == "epub") return render_epub(text, path, title);
    return false;
}

static Json::Value parse_json(const std::string& body) {
    Json::CharReaderBuilder builder;
    Json::Value value;
    std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(body.data(), body.data() + body.size(), &value, &errors))
        value = Json::objectValue;
    return value;
}

static MHD_Response* json_response(const Json::Value& value, unsigned status) {
    Json::StreamWriterBuilder builder;
    std::string body = Json::writeString(builder, value);
    MHD_Response* response = MHD_create_response_from_buffer(
        body.size(), const_cast<char*>(body.data()),
        MHD_RESPMEM_MUST_COPY);
    MHD_add_response_header(response, "Content-Type", "application/json");
    MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
    return response;
}

static MHD_Response* text_response(const std::string& body, unsigned status,
                                   const char* content_type) {
    MHD_Response* response = MHD_create_response_from_buffer(
        body.size(), const_cast<char*>(body.data()),
        MHD_RESPMEM_MUST_COPY);
    MHD_add_response_header(response, "Content-Type", content_type);
    MHD_add_response_header(response, "Access-Control-Allow-Origin", "*");
    return response;
}

static MHD_Response* error_response(const std::string& message, unsigned status) {
    Json::Value value(Json::objectValue);
    value["error"] = message;
    MHD_Response* response = json_response(value, status);
    MHD_add_response_header(response, "X-Morphix-Status",
                            std::to_string(status).c_str());
    return response;
}

static MHD_Result post_iterator(void* cls, MHD_ValueKind,
                                const char* key, const char* filename,
                                const char*, const char*, const char* data,
                                uint64_t, size_t size) {
    ConnectionInfo* info = static_cast<ConnectionInfo*>(cls);
    if (filename) {
        info->upload_filename = filename;
        bool found = false;
        for (auto& upload : info->uploads) {
            if (upload.first == filename) {
                upload.second.append(data, size);
                found = true;
                break;
            }
        }
        if (!found) info->uploads.emplace_back(filename, std::string(data, size));
        info->upload_data.append(data, size);
    } else if (key && data) {
        info->json_body += std::string(key) + "=";
        info->json_body.append(data, size);
    }
    return MHD_YES;
}

static std::vector<std::pair<std::string, std::string>>
multipart_uploads(const std::string& body, const std::string& boundary) {
    std::vector<std::pair<std::string, std::string>> uploads;
    if (boundary.empty()) return uploads;
    std::string delimiter = "--" + boundary;
    size_t position = 0;
    while ((position = body.find(delimiter, position)) != std::string::npos) {
        position += delimiter.size();
        if (body.compare(position, 2, "--") == 0) break;
        if (body.compare(position, 2, "\r\n") == 0) position += 2;
        size_t header_end = body.find("\r\n\r\n", position);
        if (header_end == std::string::npos) break;
        std::string headers = body.substr(position, header_end - position);
        size_t data_start = header_end + 4;
        size_t data_end = body.find("\r\n" + delimiter, data_start);
        if (data_end == std::string::npos) data_end = body.size();
        std::regex filename_re("filename=\"([^\"]*)\"");
        std::smatch match;
        if (std::regex_search(headers, match, filename_re)) {
            uploads.emplace_back(match[1].str(),
                                 body.substr(data_start, data_end - data_start));
        }
        position = data_end;
    }
    return uploads;
}

static std::vector<std::pair<int,int>> parse_ranges(const std::string& spec, int total) {
    std::vector<std::pair<int,int>> ranges;
    std::regex item(R"(\s*(\d+)\s*(?:-\s*(\d+)\s*)?([,;]|$))");
    auto begin = std::sregex_iterator(spec.begin(), spec.end(), item);
    auto end = std::sregex_iterator();
    for (; begin != end; ++begin) {
        int first = std::stoi(begin->str(1));
        int last = begin->str(2).empty() ? first : std::stoi(begin->str(2));
        if (first < 1 || last < first || last > total)
            throw std::invalid_argument("Range out of bounds.");
        ranges.emplace_back(first, last);
    }
    if (ranges.empty()) throw std::invalid_argument("Invalid range specification.");
    return ranges;
}

static std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') out += "'\"'\"'";
        else out += c;
    }
    return out + "'";
}

static bool split_pdf(const fs::path& source, const std::vector<std::pair<int,int>>& ranges,
                      const std::vector<fs::path>& outputs) {
    for (size_t i = 0; i < ranges.size(); ++i) {
        std::ostringstream command;
        command << "qpdf --empty --pages " << shell_quote(source.string())
                << " " << ranges[i].first;
        if (ranges[i].second != ranges[i].first)
            command << "-" << ranges[i].second;
        command << " -- " << shell_quote(outputs[i].string()) << " >/dev/null 2>&1";
        if (std::system(command.str().c_str()) != 0) return false;
    }
    return true;
}

static std::vector<std::string> heading_parts(const std::string& text) {
    std::vector<std::string> parts;
    std::istringstream in(text);
    std::string line, current;
    bool in_heading = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.front() == '#') {
            if (in_heading && !trim(current).empty()) parts.push_back(trim(current));
            current = line + "\n";
            in_heading = true;
        } else {
            current += line + "\n";
        }
    }
    if (in_heading && !trim(current).empty()) parts.push_back(trim(current));
    if (parts.empty()) parts.push_back(text);
    return parts;
}

static size_t curl_write(void* contents, size_t size, size_t count, void* user) {
    size_t total = size * count;
    static_cast<std::string*>(user)->append(static_cast<char*>(contents), total);
    return total;
}

static bool curl_post(const std::string& url,
                      const std::vector<std::string>& extra_headers,
                      const std::string& body, std::string& response,
                      std::string& error, long* http_status_out = nullptr) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        error = "Could not initialize libcurl.";
        return false;
    }
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    for (const std::string& header : extra_headers)
        headers = curl_slist_append(headers, header.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    CURLcode result = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    if (http_status_out) *http_status_out = status;
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (result != CURLE_OK) {
        error = curl_easy_strerror(result);
        return false;
    }
    if (status >= 400) {
        error = "AI provider returned HTTP " + std::to_string(status);
        return false;
    }
    return true;
}

static std::string call_claude(const std::string& key, const std::string& model,
                               const std::string& system, const std::string& prompt) {
    Json::Value body(Json::objectValue);
    body["model"] = model.empty() ? "claude-sonnet-4" : model;
    body["max_tokens"] = 4000;
    body["system"] = system;
    Json::Value message(Json::objectValue);
    message["role"] = "user";
    message["content"] = prompt;
    body["messages"].append(message);
    Json::StreamWriterBuilder builder;
    std::string request = Json::writeString(builder, body);
    std::string response, error;
    std::vector<std::string> headers = {
        "x-api-key: " + key,
        "anthropic-version: 2023-06-01"
    };
    if (!curl_post("https://api.anthropic.com/v1/messages", headers,
                    request, response, error))
        throw std::runtime_error(error);
    Json::Value parsed = parse_json(response);
    std::string text;
    for (const Json::Value& block : parsed["content"]) {
        if (block["type"].asString() == "text") text += block["text"].asString();
    }
    if (text.empty()) throw std::runtime_error("Claude returned no text.");
    return text;
}

static std::string call_gemini(const std::string& key, const std::string& model,
                               const std::string& system, const std::string& prompt,
                               bool structured_split = false) {
    std::string model_name = model.empty() ? "gemini-2.5-flash" : model;
    Json::Value body(Json::objectValue);
    Json::Value system_instruction(Json::objectValue);
    Json::Value system_part(Json::objectValue);
    system_part["text"] = system;
    system_instruction["parts"].append(system_part);
    body["systemInstruction"] = system_instruction;
    Json::Value content(Json::objectValue);
    Json::Value user_part(Json::objectValue);
    user_part["text"] = prompt;
    content["role"] = "user";
    content["parts"].append(user_part);
    body["contents"].append(content);
    if (structured_split) {
        Json::Value config(Json::objectValue);
        config["responseMimeType"] = "application/json";
        Json::Value schema(Json::objectValue);
        schema["type"] = "object";
        Json::Value sections_schema(Json::objectValue);
        sections_schema["type"] = "array";
        Json::Value item(Json::objectValue);
        item["type"] = "object";
        item["properties"]["title"]["type"] = "string";
        item["properties"]["kind"]["type"] = "string";
        item["properties"]["start_page_index"]["type"] = "integer";
        item["properties"]["filename"]["type"] = "string";
        item["required"].append("title");
        item["required"].append("kind");
        item["required"].append("start_page_index");
        sections_schema["items"] = item;
        schema["properties"]["sections"] = sections_schema;
        schema["required"].append("sections");
        config["responseSchema"] = schema;
        body["generationConfig"] = config;
    }
    Json::StreamWriterBuilder builder;
    std::string request = Json::writeString(builder, body);
    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/" +
                      model_name + ":generateContent";
    std::vector<std::string> headers = {"x-goog-api-key: " + key};
    std::string response, error;
    bool succeeded = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        response.clear();
        error.clear();
        long status = 0;
        if (curl_post(url, headers, request, response, error, &status)) {
            succeeded = true;
            break;
        }
        bool retryable = status == 429 || status == 500 || status == 502 ||
                         status == 503 || status == 504;
        if (!retryable || attempt == 2) throw std::runtime_error(error);
        std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
    }
    if (!succeeded) throw std::runtime_error(error);
    Json::Value parsed = parse_json(response);
    std::string text;
    for (const Json::Value& candidate : parsed["candidates"]) {
        for (const Json::Value& part : candidate["content"]["parts"])
            text += part["text"].asString();
    }
    if (text.empty()) throw std::runtime_error("Gemini returned no text.");
    return text;
}

static void set_job(const std::string& id, const std::string& status,
                    const std::string& phase, int pct,
                    const std::string& error = "") {
    std::lock_guard<std::mutex> lock(jobs_mutex);
    Job& job = jobs[id];
    job.status = status;
    job.phase = phase;
    job.pct = pct;
    job.error = error;
}

static std::string sanitize_pdf_filename_native(const std::string& name,
                                                const std::string& fallback);

static Json::Value parse_sections(const std::string& raw, int total) {
    std::string cleaned = raw;
    static const std::regex fence(R"(^```(?:json)?\s*|\s*```$)",
                                  std::regex::icase);
    cleaned = std::regex_replace(cleaned, fence, "");
    cleaned = trim(cleaned);
    size_t begin = cleaned.find('{');
    size_t end = cleaned.rfind('}');
    std::string json_text = begin == std::string::npos || end < begin
        ? cleaned : cleaned.substr(begin, end - begin + 1);
    Json::Value parsed = parse_json(json_text);
    Json::Value sections = parsed["sections"];
    if (!sections.isArray() && parsed.isMember("chapters"))
        sections = parsed["chapters"];
    if (!sections.isArray()) return Json::arrayValue;
    Json::Value normalized(Json::arrayValue);
    std::set<int> seen;
    for (const Json::Value& section : sections) {
        int start = section.isMember("start_page_index")
            ? section["start_page_index"].asInt()
            : section["start_unit"].asInt();
        if (start < 0 || start >= total) continue;
        if (seen.count(start)) continue;
        seen.insert(start);
        Json::Value item(Json::objectValue);
        item["title"] = section["title"].asString();
        std::string kind = section["kind"].asString();
        if (kind == "frontmatter" || kind == "chapter" || kind == "backmatter")
            item["kind"] = kind;
        else
            item["kind"] = "chapter";
        if (section.isMember("filename"))
            item["filename"] = sanitize_pdf_filename_native(
                section["filename"].asString(),
                section["title"].asString());
        item["start_unit"] = start;
        normalized.append(item);
    }
    if (!normalized.empty() &&
        normalized[0]["start_unit"].asInt() != 0) {
        Json::Value first(Json::objectValue);
        first["title"] = "Front Matter";
        first["kind"] = "frontmatter";
        first["filename"] = "00_Front_Matter";
        first["start_unit"] = 0;
        Json::Value with_first(Json::arrayValue);
        with_first.append(first);
        for (const Json::Value& section : normalized)
            with_first.append(section);
        normalized = with_first;
    }
    std::vector<Json::Value> sorted;
    for (const Json::Value& section : normalized)
        sorted.push_back(section);
    std::sort(sorted.begin(), sorted.end(),
        [](const Json::Value& a, const Json::Value& b) {
            return a["start_unit"].asInt() < b["start_unit"].asInt();
        });
    normalized = Json::arrayValue;
    for (const Json::Value& section : sorted)
        normalized.append(section);
    return normalized;
}

static std::string normalized_boundary_text(const std::string& text) {
    std::string out;
    for (unsigned char c : text) {
        if (std::isalnum(c)) out += static_cast<char>(std::tolower(c));
    }
    return out;
}

static std::string sanitize_title_native(const std::string& title) {
    std::string out;
    for (unsigned char c : title) {
        if (std::isalnum(c) || c == '-' || c == '_') out += static_cast<char>(std::tolower(c));
        else if (std::isspace(c) || c == '.') {
            if (!out.empty() && out.back() != '_') out += '_';
        }
    }
    if (out.empty()) return "section";
    if (out.size() > 60) out.resize(60);
    while (!out.empty() && (out.back() == '_' || out.back() == '-'))
        out.pop_back();
    return out.empty() ? "section" : out;
}

static std::string sanitize_pdf_filename_native(const std::string& name,
                                                const std::string& fallback) {
    std::string out;
    for (unsigned char c : name) {
        if (std::isalnum(c) || c == '-' || c == '.' || c == '_' ||
            std::isspace(c)) {
            if (std::isspace(c)) {
                if (!out.empty() && out.back() != '_') out += '_';
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    while (!out.empty() && (out.front() == '.' || out.front() == '_' ||
                            out.front() == '-' || std::isspace(out.front())))
        out.erase(out.begin());
    while (!out.empty() && (out.back() == '.' || out.back() == '_' ||
                            out.back() == '-' || std::isspace(out.back())))
        out.pop_back();
    if (out.empty()) out = sanitize_title_native(fallback);
    if (out.size() > 80) out.resize(80);
    return out;
}

static bool begins_numbered_chapter(const std::string& title) {
    static const std::regex numbered(
        R"(^\s*(?:(?:chapter)\s+)?\d{1,2}[.:)]?\s+)",
        std::regex::icase);
    return std::regex_search(title, numbered);
}

static bool is_part_heading(const std::string& title) {
    static const std::regex part(R"(^\s*(?:part|section)\s+)",
                                 std::regex::icase);
    return std::regex_search(title, part);
}

static void normalize_pdf_sections(Json::Value& sections) {
    Json::Value normalized(Json::arrayValue);
    for (const Json::Value& section : sections) {
        std::string title = section["title"].asString();
        std::string normalized_title = normalized_boundary_text(title);
        if (normalized_title == "blankpage" ||
            normalized_title == "thispageintentionallyleftblank")
            continue;
        static const std::regex continuation(
            R"(^(.*?)\s+(?:page\s+\d+|continued(?:\s+\d+)?)$)",
            std::regex::icase);
        std::smatch match;
        if (std::regex_match(title, match, continuation) &&
            !normalized.empty()) {
            std::string base = match[1].str();
            std::string previous = normalized[normalized.size() - 1]["title"].asString();
            if (normalized_boundary_text(base) ==
                normalized_boundary_text(previous))
                continue;
        }
        normalized.append(section);
    }
    if (normalized.empty()) return;
    normalized[0]["start_unit"] = 0;

    Json::ArrayIndex i = 0;
    while (i + 1 < normalized.size()) {
        std::string title = normalized[i]["title"].asString();
        std::string next_title = normalized[i + 1]["title"].asString();
        int current = normalized[i]["start_unit"].asInt();
        int following = normalized[i + 1]["start_unit"].asInt();
        bool nearby_numbered =
            normalized[i]["kind"].asString() == "chapter" &&
            normalized[i + 1]["kind"].asString() == "chapter" &&
            begins_numbered_chapter(next_title) &&
            following > current && following - current <= 2;
        if (is_part_heading(title) || nearby_numbered) {
            normalized[i + 1]["start_unit"] = current;
            normalized.removeIndex(i, nullptr);
            i = 0;
        } else {
            ++i;
        }
    }
    sections = normalized;
}

static void align_pdf_section_starts_native(Json::Value& sections,
                                             const fs::path& path) {
    std::unique_ptr<poppler::document> document(
        poppler::document::load_from_file(path.string()));
    if (!document) return;
    for (Json::ArrayIndex i = 1; i < sections.size(); ++i) {
        int start = sections[i]["start_unit"].asInt();
        int previous = sections[i - 1]["start_unit"].asInt();
        std::string title = normalized_boundary_text(
            sections[i]["title"].asString());
        if (start <= previous + 1 || title.size() < 12) continue;
        int previous_page = start - 1;
        if (previous_page < 0) continue;
        std::unique_ptr<poppler::page> page(document->create_page(previous_page));
        if (!page) continue;
        poppler::byte_array bytes = page->text().to_utf8();
        std::string text(bytes.begin(), bytes.end());
        for (char& c : text) c = static_cast<char>(std::tolower(
            static_cast<unsigned char>(c)));
        std::string page_normalized;
        for (unsigned char c : text)
            if (std::isalnum(c)) page_normalized += static_cast<char>(c);
        if (page_normalized.find(title) != std::string::npos)
            sections[i]["start_unit"] = previous_page;
    }
}

static std::vector<std::string> build_pdf_outline(const fs::path& path, int total) {
    std::vector<std::string> lines;
    std::unique_ptr<poppler::document> document(
        poppler::document::load_from_file(path.string()));
    if (!document) return lines;
    for (int i = 0; i < total; ++i) {
        std::unique_ptr<poppler::page> page(document->create_page(i));
        std::string text;
        if (page) {
            poppler::byte_array bytes = page->text(
                poppler::rectf(), poppler::page::raw_order_layout).to_utf8();
            text.append(bytes.begin(), bytes.end());
        }
        std::string compact;
        std::istringstream input(text);
        std::string line;
        int line_count = 0;
        while (std::getline(input, line) && line_count < 8) {
            line = trim(line);
            if (!line.empty()) {
                if (!compact.empty()) compact += "; ";
                compact += line;
                ++line_count;
            }
        }
        if (compact.size() > 140) compact.resize(140);
        lines.push_back("[" + std::to_string(i) + "] " + compact);
    }
    return lines;
}

static std::string truncate_outline_native(const std::vector<std::string>& lines,
                                           size_t max_chars,
                                           bool& truncated) {
    std::string text;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) text += "\n";
        text += lines[i];
    }
    if (text.size() <= max_chars) {
        truncated = false;
        return text;
    }

    truncated = true;
    const std::string marker =
        "[Outline sampled evenly across the document due to length.]";
    size_t budget = max_chars > marker.size() + 1
        ? max_chars - marker.size() - 1 : max_chars;
    size_t count = std::min(lines.size(),
        std::max<size_t>(2, lines.size() * budget / std::max<size_t>(1, text.size())));

    while (count >= 2) {
        size_t edge_count = std::min(std::max<size_t>(1, count / 10),
                                     lines.size() / 3);
        size_t middle_start = edge_count;
        size_t middle_end = lines.size() - edge_count;
        size_t middle_count = count > edge_count * 2
            ? count - edge_count * 2 : 0;
        std::vector<size_t> indexes;
        for (size_t i = 0; i < edge_count; ++i) indexes.push_back(i);
        if (middle_count == 1)
            indexes.push_back((middle_start + middle_end - 1) / 2);
        else if (middle_count > 1) {
            for (size_t i = 0; i < middle_count; ++i) {
                size_t value = middle_start +
                    (i * (middle_end - middle_start - 1) / (middle_count - 1));
                indexes.push_back(value);
            }
        }
        for (size_t i = lines.size() - edge_count; i < lines.size(); ++i)
            indexes.push_back(i);
        std::sort(indexes.begin(), indexes.end());
        indexes.erase(std::unique(indexes.begin(), indexes.end()), indexes.end());

        std::string sampled = marker;
        for (size_t i : indexes) sampled += "\n" + lines[i];
        if (sampled.size() <= max_chars) return sampled;
        if (count == 2) break;
        count = std::max<size_t>(2, std::min<size_t>(count - 1,
            count * budget / std::max<size_t>(1, sampled.size())));
    }
    return text.substr(0, max_chars);
}

static std::string build_native_split_prompt(const std::vector<std::string>& lines,
                                             int total,
                                             bool& truncated) {
    std::string outline = truncate_outline_native(lines, 40000, truncated);
    std::ostringstream prompt;
    prompt
        << "The PDF has " << total << " physical pages indexed from 0. "
        << "Split the book at chapter level so each chapter becomes its own standalone PDF.\n\n"
        << "CRITICAL: Do NOT group all front matter into one section, and do NOT group all back matter into one. "
        << "Each distinct front-matter or back-matter item must be its OWN section. "
        << "Examples: cover, copyright, dedication, table of contents, foreword, preface, acknowledgements, "
        << "appendix, glossary, bibliography, index, colophon, and about the author.\n\n"
        << "CHAPTER-LEVEL SPLITTING: Each numbered chapter must be its own section, even after a Part heading. "
        << "Do NOT group all chapters in a Part into one section. Include an immediately preceding Part page "
        << "with the first chapter rather than creating a separate Part file. "
        << "Do not split on subheadings, running headers, or contents-only entries.\n\n"
        << "BOUNDARY COORDINATES ARE ZERO-BASED PHYSICAL PDF PAGE INDICES, NOT PRINTED PAGE NUMBERS. "
        << "Copy the bracketed index of each section's first page into start_page_index.\n\n"
        << "Return STRICT JSON, no prose: "
        << "{\\\"sections\\\":[{\\\"title\\\":\\\"Human title\\\",\\\"kind\\\":"
        << "\\\"frontmatter|chapter|backmatter\\\",\\\"start_page_index\\\":0,\\\"filename\\\":\\\"ascii_name\\\"}]}\n"
        << "Starts must be strictly increasing, the first is 0, and each section ends before the next starts.\n"
        << "Front/back matter use number=null and filename prefixed 00_ or 99_. "
        << "Filenames use ASCII letters/digits/underscores only and are at most 80 characters.\n\n"
        << "OUTLINE:\n" << outline;
    return prompt.str();
}

static void run_ai_split_job(const std::string& job_id, const Record& record,
                             const std::string& provider, const std::string& key,
                             const std::string& model, const std::string& target) {
    try {
        set_job(job_id, "running", "extracting", 10);
        std::string text = extract_text(record.path, record.ext);
        bool truncated = false;
        std::string prompt_outline;
        int total = 0;
        if (record.ext == "pdf") {
            std::unique_ptr<poppler::document> document(
                poppler::document::load_from_file(record.path.string()));
            if (!document) throw std::runtime_error("Could not read PDF.");
            total = document->pages();
            if (total < 2) throw std::runtime_error("Document is too short to split into chapters.");
            auto lines = build_pdf_outline(record.path, total);
            prompt_outline = build_native_split_prompt(lines, total, truncated);
        } else {
            total = static_cast<int>(heading_parts(text).size());
            if (total < 2) throw std::runtime_error("Document is too short to split into chapters.");
            prompt_outline = "Below is an outline of a document broken into " +
                             std::to_string(total) + " units numbered from 0.\n"
                             "Identify natural section boundaries and return strict JSON:\n"
                             "{\"sections\":[{\"title\":\"string\",\"start_unit\":0}]}.\n\n"
                             "OUTLINE:\n";
            std::string sampled = text.size() > 40000
                ? text.substr(0, 20000) + "\n[...document truncated...]\n" +
                  text.substr(text.size() - 20000)
                : text;
            truncated = sampled.size() != text.size();
            prompt_outline += sampled;
        }
        if (truncated) {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            jobs[job_id].truncated = true;
        }
        set_job(job_id, "running", "analyzing", 45);
        std::string system =
            "Return only JSON with a sections array. Every section needs title, "
            "kind (frontmatter, chapter, or backmatter), and start_page_index.";
        std::string raw = provider == "claude"
            ? call_claude(key, model, system, prompt_outline)
            : call_gemini(key, model, system, prompt_outline, true);
        Json::Value sections = parse_sections(raw, total);
        if (sections.empty()) throw std::runtime_error("AI returned no valid sections.");
        if (record.ext == "pdf")
            align_pdf_section_starts_native(sections, record.path);
        normalize_pdf_sections(sections);
        set_job(job_id, "running", "splitting", 75);

        Json::Value chapters(Json::arrayValue);
        int chapter_number = 0;
        for (Json::ArrayIndex i = 0; i < sections.size(); ++i) {
            int first = sections[i]["start_unit"].asInt();
            int last = i + 1 < sections.size()
                ? sections[i + 1]["start_unit"].asInt() - 1 : total - 1;
            if (last < first) continue;
            std::string id = random_id();
            fs::path path = storage_dir / (id + "." +
                (target.empty() ? "pdf" : target));
            bool created = false;
            if (record.ext == "pdf" && (target.empty() || target == "pdf")) {
                created = split_pdf(record.path, {{first + 1, last + 1}}, {path});
            } else if (record.ext == "pdf") {
                std::unique_ptr<poppler::document> document(
                    poppler::document::load_from_file(record.path.string()));
                std::string section_text;
                if (document) {
                    for (int page_index = first;
                         page_index <= last && page_index < document->pages();
                         ++page_index) {
                        std::unique_ptr<poppler::page> page(
                            document->create_page(page_index));
                        if (!page) continue;
                        poppler::byte_array bytes = page->text().to_utf8();
                        section_text.append(bytes.begin(), bytes.end());
                        section_text += "\n";
                    }
                }
                created = render_document(section_text,
                    target, path, sections[i]["title"].asString());
            } else if (record.ext != "pdf") {
                std::string section_text;
                std::string original = extract_text(record.path, record.ext);
                auto parts = heading_parts(original);
                for (int u = first; u <= last && u < static_cast<int>(parts.size()); ++u)
                    section_text += parts[u] + "\n";
                created = render_document(section_text,
                    target.empty() ? "txt" : target, path,
                    sections[i]["title"].asString());
            }
            if (!created) continue;
            std::string title = sections[i]["title"].asString();
            std::string filename;
            if (record.ext == "pdf") {
                std::string kind = sections[i]["kind"].asString();
                int number;
                if (kind == "frontmatter") number = 0;
                else if (kind == "backmatter") number = 99;
                else number = ++chapter_number;
                int width = total >= 1000 ? 4 : 3;
                char first_text[16], last_text[16];
                std::snprintf(first_text, sizeof(first_text), "%0*d", width, first + 1);
                std::snprintf(last_text, sizeof(last_text), "%0*d", width, last + 1);
                filename = "pages_" + std::string(first_text) + "_" +
                           std::string(last_text) + "_" +
                           (number < 10 ? "0" : "") + std::to_string(number) +
                           "_" + sanitize_title_native(title) + "." +
                           (target.empty() ? "pdf" : target);
            } else {
                filename = sections[i].isMember("filename")
                    ? sanitize_pdf_filename_native(sections[i]["filename"].asString(), title)
                    : safe_stem(title);
                filename += "." + (target.empty() ? "txt" : target);
            }
            outputs[id] = {id, filename,
                           target.empty() ? "pdf" : target, path};
            Json::Value chapter(Json::objectValue);
            chapter["output_id"] = id;
            chapter["filename"] = filename;
            chapters.append(chapter);
        }
        if (chapters.empty()) throw std::runtime_error("No chapter outputs could be created.");
        {
            std::lock_guard<std::mutex> lock(jobs_mutex);
            jobs[job_id].chapters = chapters;
        }
        set_job(job_id, "done", "complete", 100);
    } catch (const std::exception& error) {
        set_job(job_id, "error", "failed", 100, error.what());
    }
}

static MHD_Result post_request_complete(void*, struct MHD_Connection*,
                                        void** con_cls,
                                        enum MHD_RequestTerminationCode) {
    if (*con_cls) {
        ConnectionInfo* info = static_cast<ConnectionInfo*>(*con_cls);
        if (info->processor) MHD_destroy_post_processor(info->processor);
        delete info;
        *con_cls = nullptr;
    }
    return MHD_YES;
}

static MHD_Result answer_request(void*,
                                 struct MHD_Connection* connection,
                                 const char* url,
                                 const char* method,
                                 const char*,
                                 const char* upload_data,
                                 size_t* upload_data_size,
                                 void** con_cls) {
    if (!*con_cls) {
        *con_cls = new ConnectionInfo();
        const char* content_type =
            MHD_lookup_connection_value(connection, MHD_HEADER_KIND, "Content-Type");
        if (content_type && std::strstr(content_type, "multipart/form-data")) {
            std::string type = content_type;
            size_t boundary = type.find("boundary=");
            if (boundary != std::string::npos) {
                std::string value = type.substr(boundary + 9);
                if (!value.empty() && value.front() == '"') {
                    value.erase(value.begin());
                    size_t end = value.find('"');
                    if (end != std::string::npos) value.resize(end);
                }
                static_cast<ConnectionInfo*>(*con_cls)->boundary = value;
            }
            static_cast<ConnectionInfo*>(*con_cls)->processor =
                MHD_create_post_processor(connection, 65536, post_iterator, *con_cls);
        }
    }

    ConnectionInfo* info = static_cast<ConnectionInfo*>(*con_cls);
    if (*upload_data_size != 0) {
        info->json_body.append(upload_data, *upload_data_size);
        if (info->processor)
            MHD_post_process(info->processor, upload_data, *upload_data_size);
        else
            info->json_body.append(upload_data, *upload_data_size);
        *upload_data_size = 0;
        return MHD_YES;
    }
    if (info->first_callback) {
        info->first_callback = false;
        return MHD_YES;
    }

    std::string path = url;
    std::string http_method = method ? method : "GET";
    MHD_Response* response = nullptr;
    unsigned status = MHD_HTTP_OK;

    if (path == "/" && http_method == "GET") {
        fs::path index = fs::current_path() / "projects/morphix.cpp/index.html";
        std::string body = read_file(index);
        if (body.empty()) body = "<!doctype html><title>Morphix C++</title><h1>Morphix C++</h1>";
        response = text_response(body, MHD_HTTP_OK, "text/html");
    } else if (path == "/api/upload-batch" && http_method == "POST") {
        Json::Value value(Json::objectValue);
        Json::Value results(Json::arrayValue);
        Json::Value errors(Json::arrayValue);
        std::vector<std::pair<std::string, std::string>> uploads =
            multipart_uploads(info->json_body, info->boundary);
        if (uploads.empty()) uploads = info->uploads;
        if (uploads.empty()) {
            response = error_response("No files in the request.", MHD_HTTP_BAD_REQUEST);
        } else {
            for (const auto& upload : uploads) {
                std::string filename = fs::path(upload.first).filename().string();
                std::string ext = extension(filename);
                if (filename.empty() || !allowed_extension(ext) || upload.second.empty()) {
                    Json::Value error(Json::objectValue);
                    error["filename"] = filename;
                    error["error"] = "Unsupported or invalid file.";
                    errors.append(error);
                    continue;
                }
                std::string id = random_id();
                fs::path destination = storage_dir / (id + "." + ext);
                if (!write_file(destination, upload.second)) {
                    Json::Value error(Json::objectValue);
                    error["filename"] = filename;
                    error["error"] = "Could not save upload.";
                    errors.append(error);
                    continue;
                }
                std::string text = extract_text(destination, ext);
                files[id] = {id, filename, ext, destination};
                Json::Value file(Json::objectValue);
                file["file_id"] = id;
                file["filename"] = filename;
                file["ext"] = ext;
                file["meta"]["units"] = static_cast<Json::UInt64>(
                    ext == "pdf" ? 0 : heading_parts(text).size());
                file["meta"]["text_chars"] = static_cast<Json::UInt64>(text.size());
                results.append(file);
            }
            value["files"] = results;
            value["errors"] = errors;
            response = json_response(value, MHD_HTTP_OK);
        }
    } else if (path == "/api/upload" && http_method == "POST") {
        std::vector<std::pair<std::string, std::string>> uploads =
            multipart_uploads(info->json_body, info->boundary);
        if (uploads.empty()) uploads = info->uploads;
        std::string filename;
        std::string upload_data;
        if (!uploads.empty()) {
            filename = fs::path(uploads.front().first).filename().string();
            upload_data = uploads.front().second;
        }
        std::string ext = extension(filename);
        if (filename.empty() || !allowed_extension(ext) || upload_data.empty()) {
            response = error_response("Unsupported or invalid file.", MHD_HTTP_BAD_REQUEST);
        } else {
            std::string id = random_id();
            fs::path destination = storage_dir / (id + "." + ext);
            if (!write_file(destination, upload_data)) {
                response = error_response("Could not save upload.", MHD_HTTP_INTERNAL_SERVER_ERROR);
            } else {
                std::string text = extract_text(destination, ext);
                files[id] = {id, filename, ext, destination};
                Json::Value value(Json::objectValue);
                value["file_id"] = id;
                value["filename"] = filename;
                value["ext"] = ext;
                value["meta"]["units"] = ext == "pdf"
                    ? static_cast<int>(std::unique_ptr<poppler::document>(
                        poppler::document::load_from_file(destination.string())
                        )->pages())
                    : static_cast<int>(heading_parts(text).size());
                value["meta"]["text_chars"] = static_cast<Json::UInt64>(text.size());
                response = json_response(value, MHD_HTTP_OK);
            }
        }
    } else if (path == "/api/convert-batch" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        std::string target = lower(request["target_format"].asString());
        Json::Value outputs_value(Json::arrayValue);
        Json::Value errors(Json::arrayValue);
        if (target != "txt" && target != "md" && target != "html" &&
            target != "pdf" && target != "docx" && target != "epub") {
            response = error_response("Unsupported target format.", MHD_HTTP_BAD_REQUEST);
        } else {
            for (const Json::Value& id_value : request["file_ids"]) {
                std::string file_id = id_value.asString();
                if (!files.count(file_id)) {
                    Json::Value error(Json::objectValue);
                    error["file_id"] = file_id;
                    error["error"] = "File not found or session expired.";
                    errors.append(error);
                    continue;
                }
                const Record& source = files[file_id];
                std::string text = extract_text(source.path, source.ext);
                std::string output_id = random_id();
                fs::path destination = storage_dir / (output_id + "." + target);
                if (!render_document(text, target, destination, safe_stem(source.filename))) {
                    Json::Value error(Json::objectValue);
                    error["file_id"] = file_id;
                    error["filename"] = source.filename;
                    error["error"] = "Conversion failed.";
                    errors.append(error);
                    continue;
                }
                std::string filename = safe_stem(source.filename) + "." + target;
                outputs[output_id] = {output_id, filename, target, destination};
                Json::Value output(Json::objectValue);
                output["output_id"] = output_id;
                output["filename"] = filename;
                outputs_value.append(output);
            }
            Json::Value value(Json::objectValue);
            value["outputs"] = outputs_value;
            value["errors"] = errors;
            response = json_response(value, MHD_HTTP_OK);
        }
    } else if (path == "/api/convert" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        std::string file_id = request["file_id"].asString();
        std::string target = lower(request["target_format"].asString());
        if (!files.count(file_id)) {
            response = error_response("File not found or session expired.", MHD_HTTP_NOT_FOUND);
        } else if (target != "txt" && target != "md" && target != "html" &&
            target != "pdf" && target != "docx" && target != "epub") {
            response = error_response("Unsupported target format.", MHD_HTTP_BAD_REQUEST);
        } else {
            const Record& source = files[file_id];
            std::string text = extract_text(source.path, source.ext);
            std::string id = random_id();
            fs::path destination = storage_dir / (id + "." + target);
            if (!render_document(text, target, destination, safe_stem(source.filename))) {
                response = error_response("Conversion failed.", MHD_HTTP_INTERNAL_SERVER_ERROR);
            } else {
                std::string filename = safe_stem(source.filename) + "." + target;
                outputs[id] = {id, filename, target, destination};
                Json::Value value(Json::objectValue);
                value["output_id"] = id;
                value["filename"] = filename;
                response = json_response(value, MHD_HTTP_OK);
            }
        }
    } else if (path == "/api/split" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        std::string file_id = request["file_id"].asString();
        std::string spec = request["spec"].asString();
        std::string target = lower(request["target_format"].asString());
        if (!files.count(file_id)) {
            response = error_response("File not found or session expired.", MHD_HTTP_NOT_FOUND);
        } else {
            const Record& source = files[file_id];
            if (source.ext == "pdf") {
                try {
                    std::unique_ptr<poppler::document> document(
                        poppler::document::load_from_file(source.path.string()));
                    if (!document) throw std::invalid_argument("Could not read PDF.");
                    auto ranges = parse_ranges(spec, document->pages());
                    std::vector<fs::path> paths;
                    Json::Value parts(Json::arrayValue);
                    for (size_t i = 0; i < ranges.size(); ++i) {
                        std::string id = random_id();
                        fs::path path = storage_dir / (id + ".pdf");
                        paths.push_back(path);
                    }
                    if (!split_pdf(source.path, ranges, paths)) {
                        response = error_response("qpdf failed to split PDF.", MHD_HTTP_INTERNAL_SERVER_ERROR);
                    } else {
                        Json::Value value(Json::objectValue);
                        for (size_t i = 0; i < ranges.size(); ++i) {
                            std::string id = random_id();
                            fs::path final_path = storage_dir / (id + ".pdf");
                            fs::rename(paths[i], final_path);
                            std::string filename = safe_stem(source.filename) +
                                                   "_part" + std::to_string(i + 1) + ".pdf";
                            outputs[id] = {id, filename, "pdf", final_path};
                            Json::Value part(Json::objectValue);
                            part["output_id"] = id;
                            part["filename"] = filename;
                            parts.append(part);
                        }
                        value["parts"] = parts;
                        response = json_response(value, MHD_HTTP_OK);
                    }
                } catch (const std::exception& error) {
                    response = error_response(error.what(), MHD_HTTP_BAD_REQUEST);
                }
            } else {
                std::string text = extract_text(source.path, source.ext);
                auto parts = heading_parts(text);
                if (parts.size() < 2) {
                    response = error_response("No top-level headings found to split by.",
                                              MHD_HTTP_BAD_REQUEST);
                } else {
                    if (target.empty()) target = "txt";
                    if (target != "txt" && target != "md" && target != "html" &&
                        target != "pdf" && target != "docx" && target != "epub")
                        target = "txt";
                    Json::Value value(Json::objectValue);
                    Json::Value results(Json::arrayValue);
                    for (size_t i = 0; i < parts.size(); ++i) {
                        std::string id = random_id();
                        fs::path path = storage_dir / (id + "." + target);
                        render_document(parts[i], target, path,
                                        safe_stem(source.filename));
                        std::string filename = safe_stem(source.filename) +
                                               "_part" + std::to_string(i + 1) + "." + target;
                        outputs[id] = {id, filename, target, path};
                        Json::Value part(Json::objectValue);
                        part["output_id"] = id;
                        part["filename"] = filename;
                        results.append(part);
                    }
                    value["parts"] = results;
                    response = json_response(value, MHD_HTTP_OK);
                }
            }
        }
    } else if (path == "/api/download-all" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        fs::path archive_path = storage_dir / ("documents_" + random_id() + ".zip");
        int error = 0;
        zip_t* archive = zip_open(archive_path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
        std::unordered_map<std::string, bool> used_names;
        if (!archive) {
            response = error_response("Could not create download archive.",
                                      MHD_HTTP_INTERNAL_SERVER_ERROR);
        } else {
            for (const Json::Value& id_value : request["output_ids"]) {
                std::string id = id_value.asString();
                if (!outputs.count(id)) continue;
                const Record& record = outputs[id];
                std::string name = record.filename;
                int suffix = 2;
                std::string candidate = name;
                while (used_names.count(candidate)) {
                    candidate = fs::path(name).stem().string() + "_" +
                                std::to_string(suffix++) +
                                fs::path(name).extension().string();
                }
                used_names[candidate] = true;
                std::string data = read_file(record.path);
                if (!zip_add_text(archive, candidate.c_str(), data)) {
                    zip_discard(archive);
                    response = error_response("Could not add output to download archive.",
                                              MHD_HTTP_INTERNAL_SERVER_ERROR);
                    break;
                }
            }
            if (!response) {
                if (zip_close(archive) != 0) {
                    zip_discard(archive);
                    response = error_response("Could not finalize download archive.",
                                              MHD_HTTP_INTERNAL_SERVER_ERROR);
                } else {
                    std::string body = read_file(archive_path);
                    response = MHD_create_response_from_buffer(
                        body.size(), const_cast<char*>(body.data()), MHD_RESPMEM_MUST_COPY);
                    MHD_add_response_header(response, "Content-Disposition",
                        "attachment; filename=\"documents.zip\"");
                    MHD_add_response_header(response, "Content-Type", "application/zip");
                }
            }
        }
    } else if (path.rfind("/api/download/output/", 0) == 0 &&
               http_method == "GET") {
        std::string id = path.substr(std::string("/api/download/output/").size());
        if (!outputs.count(id)) {
            status = MHD_HTTP_NOT_FOUND;
            response = text_response("Not found", MHD_HTTP_NOT_FOUND, "text/plain");
        } else {
            const Record& record = outputs[id];
            std::string body = read_file(record.path);
            response = MHD_create_response_from_buffer(
                body.size(), const_cast<char*>(body.data()), MHD_RESPMEM_MUST_COPY);
            MHD_add_response_header(response, "Content-Disposition",
                ("attachment; filename=\"" + record.filename + "\"").c_str());
            MHD_add_response_header(response, "Content-Type", "application/octet-stream");
        }
    } else if (path.rfind("/api/download/source/", 0) == 0 &&
               http_method == "GET") {
        std::string id = path.substr(std::string("/api/download/source/").size());
        if (!files.count(id)) {
            status = MHD_HTTP_NOT_FOUND;
            response = text_response("Not found", MHD_HTTP_NOT_FOUND, "text/plain");
        } else {
            const Record& record = files[id];
            std::string body = read_file(record.path);
            response = MHD_create_response_from_buffer(
                body.size(), const_cast<char*>(body.data()), MHD_RESPMEM_MUST_COPY);
            MHD_add_response_header(response, "Content-Disposition",
                ("attachment; filename=\"" + record.filename + "\"").c_str());
            MHD_add_response_header(response, "Content-Type", "application/octet-stream");
        }
    } else if (path == "/api/ai/toc" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        std::string file_id = request["file_id"].asString();
        std::string provider = lower(request["provider"].asString());
        std::string model = request["model"].asString();
        std::string user_key = request["api_key"].asString();
        if (!files.count(file_id)) {
            response = error_response("File not found or session expired.", MHD_HTTP_NOT_FOUND);
        } else if (provider != "claude" && provider != "gemini") {
            response = error_response("Unknown AI provider.", MHD_HTTP_BAD_REQUEST);
        } else {
            const char* env_name = provider == "claude" ? "ANTHROPIC_API_KEY"
                                                        : "GEMINI_API_KEY";
            const char* env_key = std::getenv(env_name);
            std::string key = !user_key.empty() ? user_key
                               : (env_key ? env_key : "");
            if (key.empty()) {
                response = error_response("No API key available.", MHD_HTTP_BAD_REQUEST);
            } else {
                try {
                    std::string text = extract_text(files[file_id].path,
                                                    files[file_id].ext);
                    if (text.size() > 100000) text = text.substr(0, 100000);
                    std::string prompt = "Create a detailed nested Markdown table of contents for:\n\n" + text;
                    std::string raw = provider == "claude"
                        ? call_claude(key, model, "Generate a Markdown table of contents.", prompt)
                        : call_gemini(key, model, "Generate a Markdown table of contents.", prompt);
                    std::string id = random_id();
                    fs::path destination = storage_dir / (id + ".md");
                    write_file(destination, raw);
                    std::string filename = safe_stem(files[file_id].filename) + "_TOC.md";
                    outputs[id] = {id, filename, "md", destination};
                    Json::Value value(Json::objectValue);
                    value["toc_text"] = raw;
                    value["output_id"] = id;
                    value["filename"] = filename;
                    response = json_response(value, MHD_HTTP_OK);
                } catch (const std::exception& error) {
                    response = error_response(error.what(), MHD_HTTP_BAD_GATEWAY);
                }
            }
        }
    } else if (path.rfind("/api/jobs/", 0) == 0 && http_method == "GET") {
        std::string job_id = path.substr(std::string("/api/jobs/").size());
        std::lock_guard<std::mutex> lock(jobs_mutex);
        if (!jobs.count(job_id)) {
            status = MHD_HTTP_NOT_FOUND;
            response = error_response("Job not found or expired.", MHD_HTTP_NOT_FOUND);
        } else {
            const Job& job = jobs[job_id];
            Json::Value value(Json::objectValue);
            value["status"] = job.status;
            value["phase"] = job.phase;
            value["pct"] = job.pct;
            value["truncated_outline"] = job.truncated;
            if (!job.error.empty()) value["error"] = job.error;
            if (!job.chapters.isNull()) value["chapters"] = job.chapters;
            response = json_response(value, MHD_HTTP_OK);
        }
    } else if (path == "/api/ai/split/start" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        std::string file_id = request["file_id"].asString();
        std::string provider = lower(request["provider"].asString());
        std::string model = request["model"].asString();
        std::string user_key = request["api_key"].asString();
        std::string target = lower(request["target_format"].asString());
        if (!files.count(file_id)) {
            response = error_response("File not found or session expired.", MHD_HTTP_NOT_FOUND);
        } else if (provider != "claude" && provider != "gemini") {
            response = error_response("Unknown AI provider.", MHD_HTTP_BAD_REQUEST);
        } else {
            const char* env_name = provider == "claude" ? "ANTHROPIC_API_KEY"
                                                        : "GEMINI_API_KEY";
            const char* env_key = std::getenv(env_name);
            std::string key = !user_key.empty() ? user_key
                               : (env_key ? env_key : "");
            if (key.empty()) {
                response = error_response("No API key available.", MHD_HTTP_BAD_REQUEST);
            } else {
                std::string job_id = random_id();
                {
                    std::lock_guard<std::mutex> lock(jobs_mutex);
                    jobs[job_id] = Job{};
                }
                Record record = files[file_id];
                std::thread worker(run_ai_split_job, job_id, record,
                                   provider, key, model, target);
                worker.detach();
                Json::Value value(Json::objectValue);
                value["job_id"] = job_id;
                response = json_response(value, MHD_HTTP_OK);
            }
        }
    } else if (path == "/api/ai/split" && http_method == "POST") {
        Json::Value request = parse_json(info->json_body);
        std::string file_id = request["file_id"].asString();
        std::string provider = lower(request["provider"].asString());
        std::string model = request["model"].asString();
        std::string user_key = request["api_key"].asString();
        std::string target = lower(request["target_format"].asString());
        if (!files.count(file_id)) {
            response = error_response("File not found or session expired.", MHD_HTTP_NOT_FOUND);
        } else if (provider != "claude" && provider != "gemini") {
            response = error_response("Unknown AI provider.", MHD_HTTP_BAD_REQUEST);
        } else {
            const char* env_name = provider == "claude" ? "ANTHROPIC_API_KEY"
                                                        : "GEMINI_API_KEY";
            const char* env_key = std::getenv(env_name);
            std::string key = !user_key.empty() ? user_key
                               : (env_key ? env_key : "");
            if (key.empty()) {
                response = error_response("No API key available.", MHD_HTTP_BAD_REQUEST);
            } else {
                std::string job_id = random_id();
                {
                    std::lock_guard<std::mutex> lock(jobs_mutex);
                    jobs[job_id] = Job{};
                }
                run_ai_split_job(job_id, files[file_id], provider, key, model, target);
                std::lock_guard<std::mutex> lock(jobs_mutex);
                Job& job = jobs[job_id];
                if (job.status == "error") {
                    status = MHD_HTTP_BAD_GATEWAY;
                    response = error_response(job.error, MHD_HTTP_BAD_GATEWAY);
                } else {
                    Json::Value value(Json::objectValue);
                    value["chapters"] = job.chapters;
                    value["truncated_outline"] = job.truncated;
                    response = json_response(value, MHD_HTTP_OK);
                }
            }
        }
    } else {
        response = text_response("Not found", MHD_HTTP_NOT_FOUND, "text/plain");
    }

    if (const char* status_header =
            MHD_get_response_header(response, "X-Morphix-Status")) {
        status = static_cast<unsigned>(std::atoi(status_header));
    }
    MHD_Result result = MHD_queue_response(connection, status, response);
    MHD_destroy_response(response);
    return result;
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    storage_dir = fs::temp_directory_path() / "morphix_cpp_storage";
    fs::create_directories(storage_dir);
    int port = 5034;
    if (char* env = std::getenv("PORT"); env && *env) port = std::atoi(env);

    MHD_Daemon* daemon = MHD_start_daemon(
        MHD_USE_INTERNAL_POLLING_THREAD,
        static_cast<uint16_t>(port),
        nullptr, nullptr,
        &answer_request, nullptr,
        MHD_OPTION_NOTIFY_COMPLETED, &post_request_complete, nullptr,
        MHD_OPTION_END);
    if (!daemon) {
        std::cerr << "Failed to start Morphix C++ on port " << port << "\n";
        curl_global_cleanup();
        return 1;
    }

    std::cout << "Morphix C++ listening on http://127.0.0.1:" << port << "\n";
    std::cout.flush();
    pause();
    MHD_stop_daemon(daemon);
    curl_global_cleanup();
    return 0;
}
