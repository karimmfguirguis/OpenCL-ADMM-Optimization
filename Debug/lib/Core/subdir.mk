################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../lib/Core/Assert.cpp \
../lib/Core/BoostFilesystem.cpp \
../lib/Core/CheckedCast.cpp \
../lib/Core/Error.cpp \
../lib/Core/Exception.cpp \
../lib/Core/Image.cpp \
../lib/Core/Memory.cpp \
../lib/Core/NumericException.cpp \
../lib/Core/OStream.cpp \
../lib/Core/StrError.cpp \
../lib/Core/StringUtil.cpp \
../lib/Core/Time.cpp \
../lib/Core/TimeSpan.cpp \
../lib/Core/ToString.cpp \
../lib/Core/Type.cpp \
../lib/Core/WindowsError.cpp 

C_SRCS += \
../lib/Core/StrError.c 

O_SRCS += \
../lib/Core/BoostFilesystem.o \
../lib/Core/OStream.o \
../lib/Core/StringUtil.o \
../lib/Core/ToString.o 

OBJS += \
./lib/Core/Assert.o \
./lib/Core/BoostFilesystem.o \
./lib/Core/CheckedCast.o \
./lib/Core/Error.o \
./lib/Core/Exception.o \
./lib/Core/Image.o \
./lib/Core/Memory.o \
./lib/Core/NumericException.o \
./lib/Core/OStream.o \
./lib/Core/StrError.o \
./lib/Core/StringUtil.o \
./lib/Core/Time.o \
./lib/Core/TimeSpan.o \
./lib/Core/ToString.o \
./lib/Core/Type.o \
./lib/Core/WindowsError.o 

CPP_DEPS += \
./lib/Core/Assert.d \
./lib/Core/BoostFilesystem.d \
./lib/Core/CheckedCast.d \
./lib/Core/Error.d \
./lib/Core/Exception.d \
./lib/Core/Image.d \
./lib/Core/Memory.d \
./lib/Core/NumericException.d \
./lib/Core/OStream.d \
./lib/Core/StrError.d \
./lib/Core/StringUtil.d \
./lib/Core/Time.d \
./lib/Core/TimeSpan.d \
./lib/Core/ToString.d \
./lib/Core/Type.d \
./lib/Core/WindowsError.d 

C_DEPS += \
./lib/Core/StrError.d 


# Each subdirectory must supply rules for building sources it contributes
lib/Core/%.o: ../lib/Core/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -DOMPI_SKIP_MPICXX -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -I"/home/guirgukm/Documents/GPU Lab/ADMM_Optimization/lib" -I/usr/include/mpi -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

lib/Core/%.o: ../lib/Core/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I"/home/guirgukm/Documents/GPU Lab/ADMM_Optimization/lib" -I/usr/include/mpi -O0 -g3 -Wall -c -fmessage-length=0 -DOMPI_SKIP_MPICXX -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


