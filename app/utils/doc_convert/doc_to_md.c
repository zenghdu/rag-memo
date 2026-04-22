#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

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
        if (written > 0 && written < (int)sizeof(candidate) && access(candidate, X_OK) == 0) {
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

int doc_to_md(const char *doc_path, const char *md_path, const char *html_path) {
    if (doc_path == NULL || md_path == NULL) {
        fprintf(stderr, "invalid argument: doc_path and md_path are required\n");
        return 2;
    }

    if (access(doc_path, R_OK) != 0) {
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

    char temp_html_template[] = "/tmp/doc_to_md_XXXXXX.html";
    char actual_html_path[PATH_MAX];
    int use_temp_html = 0;

    if (html_path != NULL && *html_path != '\0') {
        int written = snprintf(actual_html_path, sizeof(actual_html_path), "%s", html_path);
        if (written <= 0 || written >= (int)sizeof(actual_html_path)) {
            fprintf(stderr, "html output path is too long\n");
            return 2;
        }
    } else {
        int fd = mkstemps(temp_html_template, 5);
        if (fd < 0) {
            perror("mkstemps");
            return 2;
        }
        close(fd);
        use_temp_html = 1;
        int written = snprintf(actual_html_path, sizeof(actual_html_path), "%s", temp_html_template);
        if (written <= 0 || written >= (int)sizeof(actual_html_path)) {
            fprintf(stderr, "failed to prepare temporary html path\n");
            unlink(temp_html_template);
            return 2;
        }
    }

    char *wv_args[] = {"wvHtml", (char *)doc_path, actual_html_path, NULL};
    int ret = run_command(wv_args);
    if (ret != 0) {
        fprintf(stderr, "wvHtml failed with exit code %d\n", ret);
        if (use_temp_html) {
            unlink(actual_html_path);
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
            unlink(actual_html_path);
        }
        return ret;
    }

    if (use_temp_html) {
        unlink(actual_html_path);
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
