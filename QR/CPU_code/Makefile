PRJ    = qr
SRC    = givens.c householder.c matrix.c qr.c
INCL   = givens.h householder.h matrix.h
OBJ    = $(SRC:%.c=%.o)

CC     = gcc
CFLAGS = -O1
LFLAGS = -lm

all: $(OBJ) $(INCL)
	$(CC)  $(CFLAGS) -o $(PRJ) $(SRC) $(LFLAGS)
	

# removes files created during build
clean:
	rm -f $(OBJ) $(PRJ)

