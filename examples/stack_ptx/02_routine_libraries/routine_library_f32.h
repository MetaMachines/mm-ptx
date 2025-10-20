#pragma once

// This is meant to be added to a "RoutineIdx" enum.
// It will declare the names so that the main program
// can refer to these routines as well as other routines.
#define ROUTINE_LIBRARY_F32     \
    ROUTINE_LIBRARY_F32_FUNC_0, \
    ROUTINE_LIBRARY_F32_FUNC_1, \
    ROUTINE_LIBRARY_F32_FUNC_2 

// This is meant to be added to an initializer array list of routines.
// "routine_library_f32_0" is going to be an array of StackPtxInstruction 
// declared in the impl.h header file.
#define ROUTINE_LIBRARY_F32_INITIALIZERS                        \
    [ROUTINE_LIBRARY_F32_FUNC_0] = routine_library_f32_func_0,  \
    [ROUTINE_LIBRARY_F32_FUNC_1] = routine_library_f32_func_1,  \
    [ROUTINE_LIBRARY_F32_FUNC_2] = routine_library_f32_func_2
