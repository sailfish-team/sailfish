VERSION=20120422

# Install directory, uncomment and change to override.
# default is '/usr/local'.
# Comment out for debian packaging.
#TARGET=/usr/local
TARGET=$(DESTDIR)/usr/

# Number of jobs during compilation
NCPU=2

include build/Makefile.build

uninstall:
	rm -f $(TARGET)/lib/libcvmlcpp.a $(TARGET)/lib/libcvmlcpp.so $(TARGET)/lib/libcvmlcpp.so.$(VERSION)
	rm -rf $(TARGET)/include/cvmlcpp $(TARGET)/share/doc/cvmlcpp
	rm -rf $(TARGET)/share/cvmlcpp   $(TARGET)/include/omptl

post-install:
	mv $(TARGET)/lib/libcvmlcpp.so $(TARGET)/lib/libcvmlcpp.so.$(VERSION)
	ln -s libcvmlcpp.so.$(VERSION) $(TARGET)/lib/libcvmlcpp.so
