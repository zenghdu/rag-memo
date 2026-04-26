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

    char *pandoc_args[] = {
        "pandoc",
        "-f", "html",
        "-t", "gfm",
        "-o", (char *)md_path,
        actual_html_path,
        NULL,
    };

    ret = run_command(pandoc_args);
    if (ret != 0) {
        fprintf(stderr, "pandoc failed with exit code %d\n", ret);
        if (use_temp_html) {
            UNLINK(actual_html_path);
        }
        return ret;
    }

    if (use_temp_html) {
        UNLINK(actual_html_path);
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
