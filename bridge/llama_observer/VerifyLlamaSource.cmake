function(cmb_verify_llama_source source_dir expected_revision)
  execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${source_dir}"
    OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  if(NOT revision STREQUAL expected_revision)
    message(FATAL_ERROR "Observer v1 requires the documented llama.cpp revision")
  endif()
  execute_process(COMMAND git status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${source_dir}" OUTPUT_VARIABLE dirty
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  if(NOT dirty STREQUAL "")
    message(FATAL_ERROR "Observer v1 requires an unmodified llama.cpp checkout")
  endif()
  set(LLAMA_REV "${revision}" PARENT_SCOPE)
endfunction()
