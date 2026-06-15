"""Search tools package — backward-compatible public API."""

from app.agents.tools.search.cities import get_all_available_cities
from app.agents.tools.search.core import (
    get_property_details,
    search_properties,
    select_property,
)
from app.agents.tools.search.display import (
    _resolve_page_size,
    _resolve_page_size_from,
    _resolve_page_size_max,
)
from app.agents.tools.search.normalize import _split_amenities_by_known
from app.agents.tools.search.pagination import (
    _build_search_page_payload,
    paginate_stored_results,
    return_to_previous_results,
)
from app.agents.tools.search.selection import _resolve_property_id_from_selection

__all__ = [
    "get_all_available_cities",
    "get_property_details",
    "search_properties",
    "select_property",
    "_resolve_page_size",
    "_resolve_page_size_from",
    "_resolve_page_size_max",
    "_split_amenities_by_known",
    "_build_search_page_payload",
    "paginate_stored_results",
    "return_to_previous_results",
    "_resolve_property_id_from_selection",
]

from app.config.agent_config_loader import cfg 
from app.agents.tools.search.normalize import _split_amenity_input
from app.agents.tools.search.display import _search_display_pagination_enabled

from app.agents.tools.search.display import _search_display_mode

from app.agents.tools.search.display import _search_display_max_inline_results 
from app.agents.tools.search.core import search_properties as _search_properties_impl
from app.agents.tools.search.core import select_property as _select_property_impl
from app.agents.tools.search.core import get_property_details as _get_property_details_impl

def _sync_facade_monkeypatches_to_search_modules() -> None:
    import sys
    import app.agents.tools.search.core as _core
    import app.agents.tools.search.display as _display
    import app.agents.tools.search.pagination as _pagination

    facade = sys.modules[__name__]
    names = [
        "cfg",
        "_search_display_mode",
        "_search_display_max_inline_results",
        "_search_display_pagination_enabled",
        "_resolve_page_size",
        "_resolve_page_size_from",
        "_resolve_page_size_max",
        "_build_search_page_payload",
    ]

    for name in names:
        if hasattr(facade, name):
            value = getattr(facade, name)
            for module in (_core, _display, _pagination):
                if hasattr(module, name):
                    setattr(module, name, value)

async def search_properties(*args, **kwargs):
    _sync_facade_monkeypatches_to_search_modules()
    return await _search_properties_impl(*args, **kwargs)

async def select_property(*args, **kwargs):
    _sync_facade_monkeypatches_to_search_modules()
    return await _select_property_impl(*args, **kwargs)

async def get_property_details(*args, **kwargs):
    _sync_facade_monkeypatches_to_search_modules()
    return await _get_property_details_impl(*args, **kwargs)
