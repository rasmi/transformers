import sys

import pytest

from transformers.testing_utils import run_test_using_subprocess
from transformers.utils.import_utils import clear_import_cache, get_module_source


@run_test_using_subprocess
def test_clear_import_cache():
    """Test the clear_import_cache function."""

    # Save initial state
    initial_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
    assert len(initial_modules) > 0, "No transformers modules loaded before test"

    # Execute clear_import_cache() function
    clear_import_cache()

    # Verify modules were removed
    remaining_modules = {name: mod for name, mod in sys.modules.items() if name.startswith("transformers.")}
    assert len(remaining_modules) < len(initial_modules), "No modules were removed"

    # Import and verify module exists
    from transformers.models.auto import modeling_auto

    assert "transformers.models.auto.modeling_auto" in sys.modules
    assert modeling_auto.__name__ == "transformers.models.auto.modeling_auto"


class TestGetModuleSource:
    def test_by_dotted_name(self):
        src = get_module_source("transformers.utils.import_utils")
        assert "def get_module_source" in src

    def test_by_module_object(self):
        from transformers.utils import import_utils

        src = get_module_source(import_utils)
        assert "def get_module_source" in src

    def test_dotted_name_and_module_object_agree(self):
        from transformers.utils import import_utils

        assert get_module_source(import_utils) == get_module_source("transformers.utils.import_utils")

    def test_module_object_package_reads_init(self):
        # Packages have __path__; the helper should read their __init__.py rather than looking for `<pkg>.py`.
        import transformers.utils as utils_pkg

        src = get_module_source(utils_pkg)
        # utils/__init__.py re-exports get_module_source via the import_utils module.
        assert "from .import_utils" in src or "import_utils" in src

    def test_builtin_module_has_no_python_source(self):
        # Built-in modules (e.g. sys) have a loader, but its get_source returns None.
        with pytest.raises(FileNotFoundError, match="No source available"):
            get_module_source("sys")

    def test_unknown_module_raises_module_not_found(self):
        # find_spec returns None for a module that doesn't exist.
        with pytest.raises(ModuleNotFoundError):
            get_module_source("transformers.utils.this_module_does_not_exist")

    def test_module_object_stem_from_file(self):
        """Verify stem is derived from __file__, not __name__."""
        from transformers.models.bert import modeling_bert

        src = get_module_source(modeling_bert)
        assert "class BertModel" in src or "BertPreTrainedModel" in src
