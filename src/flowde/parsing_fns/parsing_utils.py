from pydantic import BaseModel

from flowde.parsing_fns.parsing_types import ParseType, build_partial_flowchart_schema


def get_result_structure(
    parts_to_parse: set[ParseType] | None, result_structure: type[BaseModel] | None
) -> type[BaseModel]:
    if parts_to_parse is not None and result_structure is not None:
        msg = (
            "Cannot specify both parts_to_parse and result_structure.",
            "Please specify only one of these parameters.",
        )
        raise ValueError(msg)

    if result_structure is not None:
        return result_structure

    return build_partial_flowchart_schema(parts_to_parse=parts_to_parse)
