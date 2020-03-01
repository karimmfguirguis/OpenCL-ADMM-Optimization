Include = -I. -I../lib -I../lib/viennacl/linalg -I/appl/pas/opencv/include -I/usr/local/include/opencv2 -I/usr/include/eigen3 -I/usr/include -I/usr/include/x86_64-linux-gnu -I/usr/include/mpi -I/usr/include/x86_64-linux-gnu/ -I/usr/lib/gcc/x86_64-linux-gnu/5/include


CC=g++ --std=c++0x

ODIR=../Debug
#LDIR =../lib/Core ../lib/Math ../lib/OpenCL ../lib ../lib/viennacl /usr/include/eigen3 /usr/include/opencv /usr/include/mpi /usr/include/x86_64-linux-gnu/5/include /usr/include/x86_64-linux-gnu

LDIR = -L../lib -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -L/appl/pas/opencv/lib -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu  
LIBS= -lopencv_calib3d -lOpenCL -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lboost_system -lboost_filesystem -lboost_program_options -lpthread -lboost_thread -ldl

PrePro = -D VIENNACL_WITH_OPENCL -D VIENNACL_WITH_EIGEN
#PrePro = -D VIENNACL_WITH_OPENCL -D VIENNACL_WITH_EIGEN -D VIENNACL_DEBUG_ALL


CFLAGS = -O3 -g3 -Wall


OBJ = $(patsubst %.cpp, $(ODIR)/src/%.o, $(wildcard *.cpp)) $(patsubst ../lib/Core/%.cpp, $(ODIR)/lib/Core/%.o, $(wildcard ../lib/Core/*.cpp)) $(patsubst ../lib/Math/%.cpp, $(ODIR)/lib/Math/%.o, $(wildcard ../lib/Math/*.cpp)) $(patsubst ../lib/OpenCL/%.cpp, $(ODIR)/lib/OpenCL/%.o, $(wildcard ../lib/OpenCL/*.cpp))

#$(ODIR)/lib/Core/Assert.o $(ODIR)/lib/Core/BoostFilesystem.o $(ODIR)/lib/Core/CheckedCast.o $(ODIR)/lib/Core/Error.o $(ODIR)/lib/Core/Exception.o $(ODIR)/lib/Core/Image.o $(ODIR)/lib/Core/Memory.o $(ODIR)/lib/Core/NumericException.o $(ODIR)/lib/Core/OStream.o $(ODIR)/lib/Core/StrError.o $(ODIR)/lib/Core/StringUtil.o $(ODIR)/lib/Core/Time.o $(ODIR)/lib/Core/TimeSpan.o $(ODIR)/lib/Core/ToString.o $(ODIR)/lib/Core/Type.o $(ODIR)/lib/Core/WindowsError.o $(ODIR)/lib/Math/Abs.o $(ODIR)/lib/Math/Array.o $(ODIR)/lib/Math/DiagMatrix3.o $(ODIR)/lib/Math/Float.o $(ODIR)/lib/Math/Math.o $(ODIR)/lib/Math/Vector2.o $(ODIR)/lib/Math/Vector3.o $(ODIR)/lib/OpenCL/Device.o $(ODIR)/lib/OpenCL/Error.o $(ODIR)/lib/OpenCL/Event.o $(ODIR)/lib/OpenCL/GetError.o $(ODIR)/lib/OpenCL/Program.o


.SUFFIXS: .cl

all: main

main: $(OBJ)
	$(CC) -o main $(CFLAGS) $^ $(LDIR) $(LIBS) 

$(ODIR)/src/%.o: %.cpp header.hpp
	$(CC) -c $< $(CFLAGS) -o $@ $(PrePro) $(Include)

$(ODIR)/lib/Core%.o: ../lib/Core/%.cpp
	$(CC) -c $< $(CFLAGS) -o $@ $(Include)
$(ODIR)/lib/Math/%.o: ../lib/Math/%.cpp
	$(CC) -c $< $(CFLAGS) -o $@ $(Include)
$(ODIR)/lib/OpenCL/%.o: ../lib/OpenCL/%.cpp
	$(CC) -c $< $(CFLAGS) -o $@ $(Include)


.PHONY: clean

clean:
	rm $(OBJ) main
