#-- Imported variables from top level makefile
# BIN_DIR AUX_BIN_DIR CXX CC CFLAGS CXXFLAGS LDFLAGS

##include ../Makedef

LD=$(CC)

##CFLAGS=${DEFINECFLAGS}

LIBBASE=libbase.a


.SUFFIXES:  .c .o

include Filelists.mf

all: $(LIBBASE)

$(LIBBASE): $(LIBOBJECTS)
	ar sruv $@ $(LIBOBJECTS)

#include Filegoals.mf



.PHONY:clean
clean:splintclean
	rm -f *.[oa] *.dbg.o *.inc *.aux *.dvi *.log *~


include Dependencies.mf
