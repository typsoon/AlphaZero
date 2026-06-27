#include "../../engine/game/connect4.hpp"
#include "../args_parser/inference_server_args.hpp"
#include "../model_wrapper/model_wrapper.hpp"
#include "../schema_validator/schema_validator.hpp"

#include <exception>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP(SchemaValidatorTestGroup) {
    std::shared_ptr<SchemaValidator> validator;

    void setup() {
        std::string schema_path = std::string(ALPHAZERO_REPO_ROOT) + "/game_states/connect4.json";
        validator = get_schema_validator(schema_path);
    }

    void teardown() {}
};

TEST(SchemaValidatorTestGroup, ValidPayloadIsAccepted) {
    std::string payload = R"({
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
    })";
    CHECK_TRUE(validator->is_a_valid_boardstate(payload));
}

TEST(SchemaValidatorTestGroup, InvalidPayloadIsRejected) {
    // Missing board field
    std::string payload_missing = R"({"not_board": []})";
    CHECK_FALSE(validator->is_a_valid_boardstate(payload_missing));

    // Invalid type for board
    std::string payload_wrong_type = R"({"board": "invalid"})";
    CHECK_FALSE(validator->is_a_valid_boardstate(payload_wrong_type));
}

TEST_GROUP(ModelWrapperTestGroup) {
    std::shared_ptr<ModelWrapper> wrapper;

    void setup() {
        std::string model_path =
            std::string(ALPHAZERO_REPO_ROOT) + "/inference_server/tests/payloads/dummy_model.pt";
        wrapper = create_connect4_model_wrapper(model_path, "cpu", 800, 32);
    }

    void teardown() {}
};

TEST(ModelWrapperTestGroup, ValidInferenceReturnsJSON) {
    std::string payload = R"({
        "board": [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]
    })";

    std::string result = wrapper->predict(payload);

    auto result_json = nlohmann::json::parse(result);
    CHECK_TRUE(result_json.contains("policy"));
    CHECK_EQUAL(7, result_json["policy"].size());
}

TEST(ModelWrapperTestGroup, InvalidModelPathThrows) {
    std::string bad_path = "non_existent.pt";
    bool thrown = false;
    try {
        create_connect4_model_wrapper(bad_path, "cpu", 800, 32);
    } catch (...) {
        thrown = true;
    }
    CHECK_TRUE(thrown);
}

TEST_GROUP(ArgsParserTestGroup){void setup(){} void teardown(){}};

TEST(ArgsParserTestGroup, DefaultValues) {
    InferenceServerArgs args;
    bool show_help = false;
    std::string error;
    const char *argv[] = {"server", "--network-path", "model.pt"};
    bool result = parse_inference_server_args(3, (char **)argv, args, show_help, error);
    CHECK_TRUE(result);
    CHECK_EQUAL("model.pt", args.network_path);
    CHECK_EQUAL("cuda", args.device);
    CHECK_TRUE(args.socket.find("/tmp/alphazero-inference-AZ123/model/") == 0);
    CHECK_EQUAL(800, args.mcts_search_depth);
}

TEST(ArgsParserTestGroup, CustomDepth) {
    InferenceServerArgs args;
    bool show_help = false;
    std::string error;
    const char *argv[] = {"server", "--network-path", "model.pt", "--mcts-search-depth", "400"};
    bool result = parse_inference_server_args(5, (char **)argv, args, show_help, error);
    CHECK_TRUE(result);
    CHECK_EQUAL(400, args.mcts_search_depth);
}

TEST(ArgsParserTestGroup, InvalidDepth) {
    InferenceServerArgs args;
    bool show_help = false;
    std::string error;
    const char *argv[] = {"server", "--network-path", "model.pt", "--mcts-search-depth", "abc"};
    bool result = parse_inference_server_args(5, (char **)argv, args, show_help, error);
    CHECK_FALSE(result);
}

TEST_GROUP(Connect4StaticTests){};

TEST(Connect4StaticTests, HasWinHorizontal) {
    std::vector<std::vector<int>> board(6, std::vector<int>(7, 0));
    board[0][0] = 1;
    board[0][1] = 1;
    board[0][2] = 1;
    board[0][3] = 1;
    CHECK_TRUE(Connect4::hasWin(board, 1));
    CHECK_FALSE(Connect4::hasWin(board, -1));
}

TEST(Connect4StaticTests, IsBoardFull) {
    std::vector<std::vector<int>> board(6, std::vector<int>(7, 1));
    CHECK_TRUE(Connect4::isBoardFull(board));
    board[0][3] = 0;
    CHECK_FALSE(Connect4::isBoardFull(board));
}

#include <CppUTest/MemoryLeakWarningPlugin.h>
#include <CppUTest/TestRegistry.h>

#include <cstdlib>

int main(int ac, char **av) {
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();

    CommandLineTestRunner runner(ac, av, TestRegistry::getCurrentRegistry());

    // The CommandLineTestRunner constructor installs the MemoryLeakPlugin automatically.
    // We remove it here to avoid spurious memory leak test failures with libtorch.
    TestRegistry::getCurrentRegistry()->removePluginByName(DEF_PLUGIN_MEM_LEAK);

    int res = runner.runAllTestsMain();
    std::_Exit(res);
}
