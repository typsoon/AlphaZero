#include "schema_validator.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json-schema.hpp>
#include <nlohmann/json.hpp>
#include <utility>

SchemaValidator::SchemaValidator(std::filesystem::path schema_path)
    : schema_path(std::move(schema_path)) {}

std::shared_ptr<SchemaValidator>
get_schema_validator(std::filesystem::path schema_path) {
  std::cout << schema_path << "\n";
  return std::shared_ptr<SchemaValidator>(new SchemaValidator(schema_path));
}

bool SchemaValidator::is_a_valid_boardstate(std::string body) {
  try {
    std::ifstream schema_file(schema_path);
    if (!schema_file.is_open()) {
      return false;
    }

    const nlohmann::json schema_json = nlohmann::json::parse(schema_file);
    const nlohmann::json payload_json = nlohmann::json::parse(body);

    nlohmann::json_schema::json_validator validator;
    validator.set_root_schema(schema_json);
    validator.validate(payload_json);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}
