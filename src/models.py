from dataclasses import dataclass


@dataclass
class PromptData:
    """
    Simple container for structured prompt parts used by the UI
    and visualizations.

    - role:        "I am..., You are..." system/role description
    - context:     background info / input text
    - expectations:what the user wants the model to do
    """
    role: str = ""
    context: str = ""
    expectations: str = ""

    def full_text(self) -> str:
        """Optional helper if you ever want the whole prompt as one string."""
        parts = []
        if self.role:
            parts.append(f"Role:\n{self.role}")
        if self.context:
            parts.append(f"Context:\n{self.context}")
        if self.expectations:
            parts.append(f"Expectations:\n{self.expectations}")
        return "\n\n".join(parts)
