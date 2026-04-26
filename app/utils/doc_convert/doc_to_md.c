#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <windows.h>
#define ACCESS _access
#define UNLINK _unlink
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
#else
#include <sys/wait.h>
#include <unistd.h>
#define ACCESS access
#define UNLINK unlink
#endif

#ifndef R_OK
#define R_OK 4
#endif

#ifndef X_OK
#define X_OK 1
#endif

static int file_is_readable(const char *path) {
    return path != NULL && ACCESS(path, R_OK) == 0;
}

static char *read_text_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        perror(path);
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(fp);
        return NULL;
    }

    long file_size = ftell(fp);
    if (file_size < 0) {
        perror("ftell");
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    char *buffer = (char *)malloc((size_t)file_size + 1);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }

    size_t read_size = fread(buffer, 1, (size_t)file_size, fp);
    fclose(fp);
    buffer[read_size] = '\0';

    if (out_size != NULL) {
        *out_size = read_size;
    }
    return buffer;
}

static void write_escaped_html(FILE *fp, const char *text) {
    for (const char *p = text; *p != '\0'; ++p) {
        switch (*p) {
            case '&': fputs("&amp;", fp); break;
            case '<': fputs("&lt;", fp); break;
            case '>': fputs("&gt;", fp); break;
            default: fputc(*p, fp); break;
        }
    }
}

static int ascii_case_prefix(const char *s, const char *prefix) {
    while (*prefix != '\0') {
        char a = *s;
        char b = *prefix;
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
        ++s;
        ++prefix;
    }
    return 1;
}

static int html_entity_char(const char *entity, size_t len, const char **replacement) {
    if (len == 4 && strncmp(entity, "amp;", 4) == 0) { *replacement = "&"; return 1; }
    if (len == 3 && strncmp(entity, "lt;", 3) == 0) { *replacement = "<"; return 1; }
    if (len == 3 && strncmp(entity, "gt;", 3) == 0) { *replacement = ">"; return 1; }
    if (len == 5 && strncmp(entity, "nbsp;", 5) == 0) { *replacement = " "; return 1; }
    if (len == 6 && strncmp(entity, "ldquo;", 6) == 0) { *replacement = "\""; return 1; }
    if (len == 6 && strncmp(entity, "rdquo;", 6) == 0) { *replacement = "\""; return 1; }
    if (len == 6 && strncmp(entity, "lsquo;", 6) == 0) { *replacement = "'"; return 1; }
    if (len == 6 && strncmp(entity, "rsquo;", 6) == 0) { *replacement = "'"; return 1; }
    if (len == 6 && strncmp(entity, "middot", 6) == 0) { *replacement = "·"; return 1; }
    if (len == 7 && strncmp(entity, "hellip;", 7) == 0) { *replacement = "..."; return 1; }
    if (len == 6 && strncmp(entity, "mdash;", 6) == 0) { *replacement = "—"; return 1; }
    return 0;
}

static char *strip_html_to_text(const char *html) {
    size_t len = strlen(html);
    char *out = (char *)malloc(len * 2 + 1);
    if (out == NULL) {
        return NULL;
    }

    size_t j = 0;
    int in_tag = 0;
    int pending_space = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = html[i];
        if (!in_tag && c == '<') {
            in_tag = 1;
            if (ascii_case_prefix(html + i + 1, "br") || ascii_case_prefix(html + i + 1, "/p") || ascii_case_prefix(html + i + 1, "p")) {
                if (j > 0 && out[j - 1] != '\n') {
                    out[j++] = '\n';
                }
            }
            continue;
        }
        if (in_tag) {
            if (c == '>') {
                in_tag = 0;
            }
            continue;
        }
        if (c == '&') {
            const char *semi = strchr(html + i, ';');
            if (semi != NULL) {
                const char *replacement = NULL;
                size_t ent_len = (size_t)(semi - (html + i + 1) + 1);
                if (html_entity_char(html + i + 1, ent_len, &replacement)) {
                    size_t rlen = strlen(replacement);
                    memcpy(out + j, replacement, rlen);
                    j += rlen;
                    i = (size_t)(semi - html);
                    pending_space = 0;
                    continue;
                }
            }
        }
        if (c == '\r' || c == '\t' || c == '\n' || c == ' ') {
            if (!pending_space && j > 0 && out[j - 1] != '\n') {
                out[j++] = ' ';
                pending_space = 1;
            }
            continue;
        }
        pending_space = 0;
        out[j++] = c;
    }
    out[j] = '\0';

    char *start = out;
    while (*start == ' ' || *start == '\n') start++;
    char *end = out + strlen(out);
    while (end > start && (end[-1] == ' ' || end[-1] == '\n')) {
        --end;
    }
    *end = '\0';

    if (start != out) {
        memmove(out, start, (size_t)(end - start) + 1);
    }
    return out;
}

static int is_centered_paragraph(const char *p_open_tag) {
    return strstr(p_open_tag, "text-align: center") != NULL ||
           strstr(p_open_tag, "text-align:center") != NULL ||
           strstr(p_open_tag, "align=\"center\"") != NULL ||
           strstr(p_open_tag, "align='center'") != NULL;
}

static int looks_like_section_heading(const char *text) {
    if (text == NULL || *text == '\0') {
        return 0;
    }
    if ((unsigned char)text[0] >= 0x80 && strstr(text, "、") != NULL) {
        return 1;
    }
    if (strncmp(text, "第", 3) == 0 && (strstr(text, "章") != NULL || strstr(text, "节") != NULL)) {
        return 1;
    }
    if (text[0] >= '0' && text[0] <= '9' && (strchr(text, '.') != NULL || strstr(text, "、") != NULL)) {
        return 1;
    }
    return 0;
}

static int split_section_heading(const char *text, char *heading, size_t heading_size, const char **body_start) {
    if (!looks_like_section_heading(text)) {
        return 0;
    }

    const char *split = strstr(text, "。");
    size_t punct_len = 0;
    if (split != NULL) {
        punct_len = strlen("。");
    } else {
        split = strstr(text, "：");
        if (split != NULL) {
            punct_len = strlen("：");
        } else {
            split = strstr(text, ":");
            if (split != NULL) {
                punct_len = 1;
            }
        }
    }
    if (split == NULL) {
        return 0;
    }

    size_t heading_len = (size_t)(split - text);
    if (heading_len == 0 || heading_len >= heading_size || heading_len > 80) {
        return 0;
    }

    memcpy(heading, text, heading_len);
    heading[heading_len] = '\0';

    const char *rest = split + punct_len;
    while (*rest == ' ') rest++;
    *body_start = rest;
    return 1;
}

static int clean_wv_html(const char *input_html_path, const char *output_html_path) {
    size_t size = 0;
    char *html = read_text_file(input_html_path, &size);
    if (html == NULL) {
        return -1;
    }

    FILE *out = fopen(output_html_path, "wb");
    if (out == NULL) {
        perror(output_html_path);
        free(html);
        return -1;
    }

    fputs("<!DOCTYPE html>\n<html><head><meta charset=\"UTF-8\"></head><body>\n", out);

    int block_count = 0;
    const char *cursor = html;
    while ((cursor = strstr(cursor, "<p")) != NULL) {
        const char *tag_end = strchr(cursor, '>');
        if (tag_end == NULL) {
            break;
        }
        const char *close = strstr(tag_end, "</p>");
        if (close == NULL) {
            break;
        }

        size_t open_len = (size_t)(tag_end - cursor + 1);
        char *open_tag = (char *)malloc(open_len + 1);
        if (open_tag == NULL) {
            fclose(out);
            free(html);
            return -1;
        }
        memcpy(open_tag, cursor, open_len);
        open_tag[open_len] = '\0';

        size_t inner_len = (size_t)(close - (tag_end + 1));
        char *inner = (char *)malloc(inner_len + 1);
        if (inner == NULL) {
            free(open_tag);
            fclose(out);
            free(html);
            return -1;
        }
        memcpy(inner, tag_end + 1, inner_len);
        inner[inner_len] = '\0';

        char *text = strip_html_to_text(inner);
        int centered = is_centered_paragraph(open_tag);
        free(open_tag);
        free(inner);

        if (text != NULL && *text != '\0') {
            if (block_count == 0) {
                fprintf(out, "<h1>");
                write_escaped_html(out, text);
                fprintf(out, "</h1>\n");
                block_count++;
            } else if (centered) {
                fprintf(out, "<h2>");
                write_escaped_html(out, text);
                fprintf(out, "</h2>\n");
                block_count++;
            } else {
                char heading[256];
                const char *body_start = NULL;
                if (split_section_heading(text, heading, sizeof(heading), &body_start)) {
                    fprintf(out, "<h3>");
                    write_escaped_html(out, heading);
                    fprintf(out, "</h3>\n");
                    if (body_start != NULL && *body_start != '\0') {
                        fprintf(out, "<p>");
                        write_escaped_html(out, body_start);
                        fprintf(out, "</p>\n");
                    }
                    block_count++;
                } else {
                    fprintf(out, "<p>");
                    write_escaped_html(out, text);
                    fprintf(out, "</p>\n");
                    block_count++;
                }
            }
        }
        free(text);
        cursor = close + 4;
    }

    fputs("</body></html>\n", out);
    fclose(out);
    free(html);
    return 0;
}

#ifdef _WIN32
static int command_exists(const char *cmd) {
    if (cmd == NULL || *cmd == '\0') {
        return 0;
    }

    char resolved[PATH_MAX];
    const char *extensions[] = {NULL, ".exe", ".bat", ".cmd", ".com"};
    size_t ext_count = sizeof(extensions) / sizeof(extensions[0]);

    for (size_t i = 0; i < ext_count; ++i) {
        DWORD len = SearchPathA(NULL, cmd, extensions[i], (DWORD)sizeof(resolved), resolved, NULL);
        if (len > 0 && len < sizeof(resolved)) {
            return 1;
        }
    }

    return 0;
}

static void append_quoted_arg(char *buffer, size_t size, const char *arg) {
    if (buffer == NULL || size == 0 || arg == NULL) {
        return;
    }

    size_t len = strlen(buffer);
    if (len >= size - 1) {
        return;
    }

    if (len > 0) {
        _snprintf(buffer + len, size - len, " ");
        len = strlen(buffer);
    }

    _snprintf(buffer + len, size - len, "\"");
    len = strlen(buffer);

    for (const char *p = arg; *p != '\0' && len < size - 2; ++p) {
        if (*p == '\"') {
            if (len < size - 3) {
                buffer[len++] = '\\';
                buffer[len++] = '\"';
            }
        } else {
            buffer[len++] = *p;
        }
    }

    if (len < size - 2) {
        buffer[len++] = '\"';
    }
    buffer[len] = '\0';
}

static int run_command(char *const argv[]) {
    if (argv == NULL || argv[0] == NULL) {
        fprintf(stderr, "invalid command\n");
        return -1;
    }

    char command_line[8192] = {0};
    for (int i = 0; argv[i] != NULL; ++i) {
        append_quoted_arg(command_line, sizeof(command_line), argv[i]);
    }

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    char mutable_cmd[8192];
    _snprintf(mutable_cmd, sizeof(mutable_cmd), "%s", command_line);
    mutable_cmd[sizeof(mutable_cmd) - 1] = '\0';

    BOOL ok = CreateProcessA(
        NULL,
        mutable_cmd,
        NULL,
        NULL,
        FALSE,
        0,
        NULL,
        NULL,
        &si,
        &pi
    );

    if (!ok) {
        fprintf(stderr, "CreateProcess failed for %s (error=%lu)\n", argv[0], GetLastError());
        return -1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
        fprintf(stderr, "GetExitCodeProcess failed for %s (error=%lu)\n", argv[0], GetLastError());
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        return -1;
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)exit_code;
}

static int make_temp_html_path(char *buffer, size_t size) {
    char temp_dir[PATH_MAX];
    DWORD dir_len = GetTempPathA((DWORD)sizeof(temp_dir), temp_dir);
    if (dir_len == 0 || dir_len >= sizeof(temp_dir)) {
        fprintf(stderr, "GetTempPath failed (error=%lu)\n", GetLastError());
        return -1;
    }

    char temp_file[PATH_MAX];
    UINT unique = GetTempFileNameA(temp_dir, "d2m", 0, temp_file);
    if (unique == 0) {
        fprintf(stderr, "GetTempFileName failed (error=%lu)\n", GetLastError());
        return -1;
    }

    if (_snprintf(buffer, size, "%s.html", temp_file) < 0 || strlen(buffer) >= size) {
        fprintf(stderr, "temporary html path is too long\n");
        UNLINK(temp_file);
        return -1;
    }

    UNLINK(temp_file);
    return 0;
}
#else
static int command_exists(const char *cmd) {
    if (cmd == NULL || *cmd == '\0') {
        return 0;
    }

    const char *path_env = getenv("PATH");
    if (path_env == NULL) {
        return 0;
    }

    char *path_copy = strdup(path_env);
    if (path_copy == NULL) {
        return 0;
    }

    int found = 0;
    char *saveptr = NULL;
    for (char *dir = strtok_r(path_copy, ":", &saveptr);
         dir != NULL;
         dir = strtok_r(NULL, ":", &saveptr)) {
        char candidate[PATH_MAX];
        int written = snprintf(candidate, sizeof(candidate), "%s/%s", dir, cmd);
        if (written > 0 && written < (int)sizeof(candidate) && ACCESS(candidate, X_OK) == 0) {
            found = 1;
            break;
        }
    }

    free(path_copy);
    return found;
}

static int run_command(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return -1;
    }

    if (pid == 0) {
        execvp(argv[0], argv);
        perror(argv[0]);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    fprintf(stderr, "command terminated abnormally: %s\n", argv[0]);
    return -1;
}

static int make_temp_html_path(char *buffer, size_t size) {
    char temp_html_template[] = "/tmp/doc_to_md_XXXXXX.html";
    int fd = mkstemps(temp_html_template, 5);
    if (fd < 0) {
        perror("mkstemps");
        return -1;
    }
    close(fd);

    int written = snprintf(buffer, size, "%s", temp_html_template);
    if (written <= 0 || written >= (int)size) {
        fprintf(stderr, "failed to prepare temporary html path\n");
        UNLINK(temp_html_template);
        return -1;
    }

    return 0;
}
#endif

int doc_to_md(const char *doc_path, const char *md_path, const char *html_path) {
    if (doc_path == NULL || md_path == NULL) {
        fprintf(stderr, "invalid argument: doc_path and md_path are required\n");
        return 2;
    }

    if (!file_is_readable(doc_path)) {
        fprintf(stderr, "input file is not readable: %s\n", doc_path);
        return 2;
    }

    if (!command_exists("wvHtml")) {
        fprintf(stderr, "missing dependency: wvHtml not found in PATH\n");
        return 127;
    }

    if (!command_exists("pandoc")) {
        fprintf(stderr, "missing dependency: pandoc not found in PATH\n");
        return 127;
    }

    char actual_html_path[PATH_MAX];
    int use_temp_html = 0;

    if (html_path != NULL && *html_path != '\0') {
        int written = snprintf(actual_html_path, sizeof(actual_html_path), "%s", html_path);
        if (written <= 0 || written >= (int)sizeof(actual_html_path)) {
            fprintf(stderr, "html output path is too long\n");
            return 2;
        }
    } else {
        if (make_temp_html_path(actual_html_path, sizeof(actual_html_path)) != 0) {
            return 2;
        }
        use_temp_html = 1;
    }

    char *wv_args[] = {"wvHtml", (char *)doc_path, actual_html_path, NULL};
    int ret = run_command(wv_args);
    if (ret != 0) {
        fprintf(stderr, "wvHtml failed with exit code %d\n", ret);
        if (use_temp_html) {
            UNLINK(actual_html_path);
        }
        return ret;
    }

    char cleaned_html_path[PATH_MAX];
    int use_temp_cleaned_html = 0;
    if (make_temp_html_path(cleaned_html_path, sizeof(cleaned_html_path)) != 0) {
        if (use_temp_html) {
            UNLINK(actual_html_path);
        }
        return 2;
    }
    use_temp_cleaned_html = 1;

    if (clean_wv_html(actual_html_path, cleaned_html_path) != 0) {
        fprintf(stderr, "failed to clean intermediate html\n");
        if (use_temp_html) {
            UNLINK(actual_html_path);
        }
        if (use_temp_cleaned_html) {
            UNLINK(cleaned_html_path);
        }
        return 2;
    }

    char *pandoc_args[] = {
        "pandoc",
        "-f", "html",
        "-t", "gfm",
        "--wrap=none",
        "-o", (char *)md_path,
        cleaned_html_path,
        NULL,
    };

    ret = run_command(pandoc_args);
    if (ret != 0) {
        fprintf(stderr, "pandoc failed with exit code %d\n", ret);
        if (use_temp_html) {
            UNLINK(actual_html_path);
        }
        if (use_temp_cleaned_html) {
            UNLINK(cleaned_html_path);
        }
        return ret;
    }

    if (use_temp_html) {
        UNLINK(actual_html_path);
    }
    if (use_temp_cleaned_html) {
        UNLINK(cleaned_html_path);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: %s <input.doc> <output.md> [output.html]\n", argv[0]);
        return 1;
    }

    const char *doc_path = argv[1];
    const char *md_path = argv[2];
    const char *html_path = argc == 4 ? argv[3] : NULL;

    int ret = doc_to_md(doc_path, md_path, html_path);
    if (ret == 0) {
        printf("Conversion succeeded: %s -> %s\n", doc_path, md_path);
        if (html_path != NULL && *html_path != '\0') {
            printf("Intermediate HTML saved to: %s\n", html_path);
        }
    }

    return ret;
}
