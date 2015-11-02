# Automatically generated makefile, created by the Introjucer
# Don't edit this file! Your changes will be overwritten when you re-save the Introjucer project!

# (this disables dependency generation if multiple architectures are set)
DEPFLAGS := $(if $(word 2, $(TARGET_ARCH)), , -MMD)

ifndef CONFIG
  CONFIG=Debug
endif

ifeq ($(CONFIG),Debug)
  BINDIR := build
  LIBDIR := build
  OBJDIR := build/intermediate/Debug
  OUTDIR := build

  ifeq ($(TARGET_ARCH),)
    TARGET_ARCH := -march=native
  endif

  CPPFLAGS := $(DEPFLAGS) -std=c++11 -D "LINUX=1" -D "DEBUG=1" -D "_DEBUG=1" -D "TINYRNN_OPENCL_ACCELERATION=1" -D "JUCER_LINUX_MAKE_4253FF4F=1" -D "JUCE_APP_VERSION=0.1.0" -D "JUCE_APP_VERSION_HEX=0x100" -I /usr/include -I /usr/include/freetype2 -I ../Introjucer/JuceLibraryCode
  CFLAGS += $(CPPFLAGS) $(TARGET_ARCH) -g -ggdb -O0
  CXXFLAGS += $(CFLAGS)
  LDFLAGS += $(TARGET_ARCH) -L$(BINDIR) -L$(LIBDIR) -L/usr/X11R6/lib/ 

  TARGET := TinyRNN
  BLDCMD = $(CXX) -o $(OUTDIR)/$(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(TARGET_ARCH)
  CLEANCMD = rm -rf $(OUTDIR)/$(TARGET) $(OBJDIR)
endif

ifeq ($(CONFIG),Release)
  BINDIR := build
  LIBDIR := build
  OBJDIR := build/intermediate/Release
  OUTDIR := build

  ifeq ($(TARGET_ARCH),)
    TARGET_ARCH := -march=native
  endif

  CPPFLAGS := $(DEPFLAGS) -std=c++11 -D "LINUX=1" -D "NDEBUG=1" -D "TINYRNN_OPENCL_ACCELERATION=1" -D "JUCER_LINUX_MAKE_4253FF4F=1" -D "JUCE_APP_VERSION=0.1.0" -D "JUCE_APP_VERSION_HEX=0x100" -I /usr/include -I /usr/include/freetype2 -I ../Introjucer/JuceLibraryCode
  CFLAGS += $(CPPFLAGS) $(TARGET_ARCH) -O3
  CXXFLAGS += $(CFLAGS)
  LDFLAGS += $(TARGET_ARCH) -L$(BINDIR) -L$(LIBDIR) -fvisibility=hidden -L/usr/X11R6/lib/ 

  TARGET := TinyRNN
  BLDCMD = $(CXX) -o $(OUTDIR)/$(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(TARGET_ARCH)
  CLEANCMD = rm -rf $(OUTDIR)/$(TARGET) $(OBJDIR)
endif

OBJECTS := \
  $(OBJDIR)/pugixml_a991b903.o \
  $(OBJDIR)/Main_4c3b61fc.o \
  $(OBJDIR)/LayerTests_c825a993.o \
  $(OBJDIR)/NetworkTests_5f00a1d6.o \
  $(OBJDIR)/NeuronTests_7c250c55.o \
  $(OBJDIR)/TrainingTests_ca13a64e.o \
  $(OBJDIR)/SerializationTests_b9cd27d4.o \

.PHONY: clean

$(OUTDIR)/$(TARGET): $(OBJECTS) $(RESOURCES)
	@echo Linking TinyRNN
	-@mkdir -p $(BINDIR)
	-@mkdir -p $(LIBDIR)
	-@mkdir -p $(OUTDIR)
	@$(BLDCMD)

clean:
	@echo Cleaning TinyRNN
	@$(CLEANCMD)

strip:
	@echo Stripping TinyRNN
	-@strip --strip-unneeded $(OUTDIR)/$(TARGET)

$(OBJDIR)/pugixml_a991b903.o: ../../Tests/ThirdParty/pugixml.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling pugixml.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

$(OBJDIR)/Main_4c3b61fc.o: ../../Tests/Main.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling Main.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

$(OBJDIR)/LayerTests_c825a993.o: ../../Tests/LayerTests.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling LayerTests.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

$(OBJDIR)/NetworkTests_5f00a1d6.o: ../../Tests/NetworkTests.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling NetworkTests.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

$(OBJDIR)/NeuronTests_7c250c55.o: ../../Tests/NeuronTests.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling NeuronTests.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

$(OBJDIR)/TrainingTests_ca13a64e.o: ../../Tests/TrainingTests.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling TrainingTests.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

$(OBJDIR)/SerializationTests_b9cd27d4.o: ../../Tests/SerializationTests.cpp
	-@mkdir -p $(OBJDIR)
	@echo "Compiling SerializationTests.cpp"
	@$(CXX) $(CXXFLAGS) -o "$@" -c "$<"

-include $(OBJECTS:%.o=%.d)
