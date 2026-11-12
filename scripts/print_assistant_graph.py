"""
Print the assistant graph (app/infrastructure/llm/graphs/assistant.py) as Mermaid text.

No model, index or network is used: the graph is built with stand-ins, since
only its nodes and edges are drawn.

Usage (from the repository root):
    python -m scripts.print_assistant_graph
"""

from app.infrastructure.llm.graphs.assistant import LangGraphAssistant


class _StandIn:
    """Takes the place of the chat model, the search service and the agent."""

    def with_structured_output(self, schema, **kwargs):
        return self


def assistant_graph_mermaid() -> str:
    """The Mermaid text of the assistant graph."""
    stand_in = _StandIn()
    return LangGraphAssistant(model=stand_in, search_service=stand_in, agent=stand_in).mermaid()


if __name__ == "__main__":
    print(assistant_graph_mermaid())
