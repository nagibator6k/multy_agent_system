from typing import Any, Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable[..., Any],
        description: str,
    ) -> None:
        self._tools[name] = {
            "function": function,
            "description": description,
        }

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered.")

        return self._tools[name]["function"]

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        return tool(**kwargs)

    def list_tools(self) -> dict[str, str]:
        return {
            name: data["description"]
            for name, data in self._tools.items()
        }


def create_registry() -> ToolRegistry:
    from tools.knowledge import search_knowledge
    from tools.math_tools import solve_quadratic

    registry = ToolRegistry()

    registry.register(
        name="search_knowledge",
        function=search_knowledge,
        description=(
            "Search the educational knowledge base "
            "for relevant information."
        ),
    )

    registry.register(
        name="solve_quadratic",
        function=solve_quadratic,
        description=(
            "Solve a quadratic equation ax² + bx + c = 0 "
            "and return the discriminant and roots."
        ),
    )

    return registry


tool_registry = create_registry()