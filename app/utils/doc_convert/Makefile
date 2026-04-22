CC=gcc
CFLAGS=-O2 -Wall -Wextra
TARGET=doc_to_md
SRC=doc_to_md.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
