from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    tokens = session.dataframe["unigrams"].tolist()

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert tokens == expected_tokens
    assert res["total_documents"] == 8


def test_list_files_tool():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n": 3})

    assert res["n_rows"] == 8
    assert res["inspected_rows"] == 3
    assert len(res["head"]) == 3
    assert "text" in res["columns"]
