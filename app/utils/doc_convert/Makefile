CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
EXE_EXT ?=
TARGET = doc_to_md$(EXE_EXT)
SRC = doc_to_md.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f doc_to_md doc_to_md.exe
