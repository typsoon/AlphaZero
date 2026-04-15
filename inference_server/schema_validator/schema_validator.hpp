#ifndef ALPHAZERO_INFERENCE_SERVER_SCHEMA_VALIDATOR
#define ALPHAZERO_INFERENCE_SERVER_SCHEMA_VALIDATOR

#include <filesystem>
#include <memory>
#include <string>

class SchemaValidator {
  std::filesystem::path schema_path;
  SchemaValidator(std::filesystem::path schema_path);
  friend std::shared_ptr<SchemaValidator>
  get_schema_validator(std::filesystem::path schema_path);

public:
  bool is_a_valid_boardstate(std::string body);
};

std::shared_ptr<SchemaValidator>
get_schema_validator(std::filesystem::path schema_path);

#endif
